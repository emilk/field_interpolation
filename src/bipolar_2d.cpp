#include "sdf_field.hpp"

#include <random>

#include <visit_struct/visit_struct.hpp>

#include "configuru_extensions.hpp"
#include "imgui_intro.hpp"

#include <configuru.hpp>
#include <emath/aabb.hpp>
#include <emath/math.hpp>
#include <emath/vec2.hpp>
#include <emilib/dual.hpp>
#include <emilib/file_system.hpp>
#include <emilib/gl_lib.hpp>
#include <emilib/imgui_helpers.hpp>
#include <emilib/irange.hpp>
#include <emilib/marching_squares.hpp>
#include <emilib/strprintf.hpp>
#include <emilib/tga.hpp>
#include <emilib/timer.hpp>
#include <loguru.hpp>

#include "gui.hpp"

namespace bipolar_2d {

using namespace emath;
using namespace emilib;

using Matrixf = Matrix<float>;
using ImageRGBA = Matrix<RGBA>;

struct Options
{
	// Input equation: the cell at position X pulls towards in_values[X] with the force in_weights[X]
	Matrixf in_values;
	Matrixf in_weights;

	// Prior for C1 smoothness constraint, i.e.  f[x-1] - 2 * f[x] + f[x+1] = 0
	float   smoothness                  = 1.0f;

	// How strongly each cell pulls towards zero (for X where in_weights[X] == 0).
	float   regularization              = 1e-3f;

	// If false, constraints only act to repel from zero
	bool bilateral = false;

	// Torus world?
	bool wrapping = true;

	int cg_iterations = 5;

	Options()
	{
		int width = 96;
		int height = 64;
		in_values = Matrixf(width, height, 0.0);
		in_weights = Matrixf(width, height, 0.0);

		// Set two values:
		in_values(width  * 1 / 3, height / 2) = +2.0;
		in_weights(width * 1 / 3, height / 2) =  1.0;
		in_values(width  * 2 / 3, height / 2) = -2.0;
		in_weights(width * 2 / 3, height / 2) =  1.0;
	}
};

} // namespace bipolar_2d

VISITABLE_STRUCT(bipolar_2d::Options, in_values, in_weights, smoothness, regularization, bilateral, wrapping, cg_iterations);

namespace bipolar_2d {

struct Result
{
	fi::LinearEquation eq;
	Matrixf            field;
	std::string        log; // Log captured during solving
};

void log_errors(const std::string& msg)
{
	LOG_F(ERROR, "%s", msg.c_str());
}

fi::LinearEquation generate_equation(const Options& options, const Matrixf& last_solution)
{
	LOG_SCOPE_F(1, "generate_equation");

	CHECK_EQ_F(options.in_values.width(), options.in_weights.width());
	CHECK_EQ_F(options.in_values.height(), options.in_weights.height());
	CHECK_EQ_F(options.in_values.height(), last_solution.height());
	CHECK_EQ_F(options.in_values.height(), last_solution.height());

	const int width = options.in_values.width();
	const int height = options.in_values.height();

	const auto coord = [=](int x, int y) {
		x = (x + width)  % width;
		y = (y + height) % height;
		return y * width + x;
	};

	fi::LinearEquation eq;

	for (const int y : irange(height)) {
		for (const int x : irange(width)) {
			if (options.wrapping || (0 < x && x < width - 1)) {
				add_equation(&eq, fi::Weight{options.smoothness}, fi::Rhs{0.0f}, {
					{coord(x-1, y), +1.0f},
					{coord(x,   y), -2.0f},
					{coord(x+1, y), +1.0f},
				});
			}

			if (options.wrapping || (0 < y && y < height - 1)) {
				add_equation(&eq, fi::Weight{options.smoothness}, fi::Rhs{0.0f}, {
					{coord(x, y-1), +1.0f},
					{coord(x, y),   -2.0f},
					{coord(x, y+1), +1.0f},
				});
			}
		}
	}

	for (const int i : indices(options.in_values)) {
		float v = options.in_values[i];
		float w = options.in_weights[i];
		if (w <= 0.0) {
			v = 0.0f;
			w = options.regularization;
		}

		const bool add_constraint =
			options.bilateral
			|| v == 0.0f
			|| std::fabs(last_solution[i]) <= std::fabs(v);

		if (add_constraint) {
			fi::add_equation(&eq, fi::Weight{w}, fi::Rhs{v}, {{i, 1.0f}});
		}
	}

	return eq;
}

void ram_logger(void* user_data, const loguru::Message& msg)
{
	std::string* log = reinterpret_cast<std::string*>(user_data);
	*log += emilib::strprintf("%s%s%s\n", msg.indentation, msg.prefix, msg.message);
}

RGBA as_color(float field)
{
	if (std::fabs(field) < 1.0f) {
		return RGBA{32, 32, 32, 255};
	};
	if (field > 0.0f) {
		if (field >= 2.0f) {
			return RGBA{255, 0, 0, 255};
		} else {
			return RGBA{128, 0, 0, 255};
		}
	} else {
		if (std::fabs(field) >= 2.0f) {
			return RGBA{64, 64, 255, 255};
		} else {
			return RGBA{32, 32, 128, 255};
		}
	}
}

ImageRGBA as_image(const Matrixf& field)
{
	auto img = ImageRGBA(field.width(), field.height());
	for (const auto i : indices(field)) {
		img[i] = as_color(field[i]);
	}
	return img;
}

Result generate(const Options& options, Matrixf last_solution)
{
	const bool first_time = last_solution.empty();

	const int width = options.in_values.width();
	const int height = options.in_values.height();
	last_solution.resize(width, height, 0.0f);

	Result result;
	loguru::add_callback("ram_logger", ram_logger, &result.log, loguru::Verbosity_MAX);

	result.eq = generate_equation(options, last_solution);

	if (first_time) {
		result.field = Matrixf(width, height, fi::solve_sparse_linear_exact(result.eq, width * height));
	} else {
		// result.field = Matrixf(width, height, fi::solve_sparse_linear_fast(result.eq, width * height));
		result.field = Matrixf(width, height, fi::solve_sparse_linear_with_guess(result.eq, last_solution.as_vec(), options.cg_iterations, 0.0f));
	}

	loguru::remove_callback("ram_logger");
	return result;
}

bool show_options(Options* options)
{
	bool changed = false;

	int w = options->in_values.width();
	int h = options->in_values.height();

	changed |= ImGui::SliderInt("Width", &w, 16, 128);
	// ImGui::SameLine();
	changed |= ImGui::SliderInt("Height", &h, 16, 128);

	options->in_values.resize(w, h);
	options->in_weights.resize(w, h);

	changed |= ImGui::SliderFloat("smoothness",     &options->smoothness,     0.0f, 10.0f, "%.3f", 4);
	changed |= ImGui::SliderFloat("regularization", &options->regularization, 0.0f,  1.0f, "%.3f", 4);
	changed |= ImGui::Checkbox("bilateral", &options->bilateral);
	changed |= ImGui::Checkbox("wrapping", &options->wrapping);
	changed |= ImGui::SliderInt("cg_iterations", &options->cg_iterations, 1,  50);

	return changed;
}

void paint_grid_centers(const Options& options, ImVec2 canvas_pos, ImVec2 canvas_size)
{
	ImDrawList* draw_list = ImGui::GetWindowDrawList();

	const auto w = options.in_values.width();
	const auto h = options.in_values.height();

	if (w * h < 64 * 64) {
		for (const int yi : irange(h)) {
			for (const int xi : irange(w)) {
				// TODO: remap
				const float x = static_cast<float>(xi) / (w - 1.0f);
				const float y = static_cast<float>(yi) / (h - 1.0f);
				const float center_x = canvas_pos.x + canvas_size.x * x;
				const float center_y = canvas_pos.y + canvas_size.y * y;
				draw_list->AddCircleFilled({center_x, center_y}, 1, ImColor(1.0f, 1.0f, 1.0f, 0.25f), 4);
			}
		}
	}
}

Vec2f remap_pos(
	const Vec2f& in_pos,
	const Vec2i& in_size,
	const AABB2f& out_rect)
{
	return Vec2f{
		emath::remap(in_pos.x, 0.0f, in_size.x - 0.5f, out_rect.min().x, out_rect.max().x),
		emath::remap(in_pos.y, 0.0f, in_size.y - 0.5f, out_rect.min().y, out_rect.max().y),
	};
}

void paint_outline(
	const int                 w,
	const int                 h,
	const std::vector<float>& lines,
	const ImVec2              canvas_pos,
	const ImVec2              canvas_size,
	const ImColor             color,
	const bool                paint_normals)
{
	ImDrawList* draw_list = ImGui::GetWindowDrawList();

	CHECK_F(lines.size() % 4 == 0);

	for (int i = 0; i < lines.size(); i += 4) {
		float x0 = lines[i + 0];
		float y0 = lines[i + 1];
		float x1 = lines[i + 2];
		float y1 = lines[i + 3];

		x0 = emath::remap(x0, 0.0f, w - 0.5f, canvas_pos.x, canvas_pos.x + canvas_size.x);
		y0 = emath::remap(y0, 0.0f, h - 0.5f, canvas_pos.y, canvas_pos.y + canvas_size.y);
		x1 = emath::remap(x1, 0.0f, w - 0.5f, canvas_pos.x, canvas_pos.x + canvas_size.x);
		y1 = emath::remap(y1, 0.0f, h - 0.5f, canvas_pos.y, canvas_pos.y + canvas_size.y);

		draw_list->AddLine({x0, y0}, {x1, y1}, color);

		if (paint_normals) {
			float cx = (x0 + x1) / 2;
			float cy = (y0 + y1) / 2;
			float dx = (x1 - x0);
			float dy = (y1 - y0);
			float norm = 10 / std::hypot(dx, dy);
			dx *= norm;
			dy *= norm;
			draw_list->AddLine({cx, cy}, {cx + dy, cy - dx}, ImColor(0.0f, 1.0f, 0.0f, 1.0f));
		}
	}
}

// ----------------------------------------------------------------------------

Matrixf wrapping_bicubic_upsample(const Matrixf& small, int upsample)
{
	if (upsample <= 1) { return small; }

	CHECK_GT_F(upsample, 1);

	const auto value_at = [&](int sx, int sy)
	{
		sx = (sx + small.width())  % small.width();
		sy = (sy + small.height()) % small.height();
		return small(sx, sy);
	};

	const size_t large_width  = upsample * small.width()  - upsample + 1;
	const size_t large_height = upsample * small.height() - upsample + 1;
	auto large = Matrixf(large_width, large_height);

	for (int ly = 0; ly < large_height; ++ly) {
		for (int lx = 0; lx < large_width; ++lx) {
			int sx = lx / upsample;
			int sy = ly / upsample;

			float tx = static_cast<float>(lx % upsample) / static_cast<float>(upsample);
			float ty = static_cast<float>(ly % upsample) / static_cast<float>(upsample);

			float local[4][4] {
				{ value_at(sx - 1, sy - 1), value_at(sx + 0, sy - 1), value_at(sx + 1, sy - 1), value_at(sx + 2, sy - 1) },
				{ value_at(sx - 1, sy + 0), value_at(sx + 0, sy + 0), value_at(sx + 1, sy + 0), value_at(sx + 2, sy + 0) },
				{ value_at(sx - 1, sy + 1), value_at(sx + 0, sy + 1), value_at(sx + 1, sy + 1), value_at(sx + 2, sy + 1) },
				{ value_at(sx - 1, sy + 2), value_at(sx + 0, sy + 2), value_at(sx + 1, sy + 2), value_at(sx + 2, sy + 2) },
			};

			large(lx, ly) =  emath::catmull_rom(ty,
				emath::catmull_rom(tx, local[0][0], local[0][1], local[0][2], local[0][3]),
				emath::catmull_rom(tx, local[1][0], local[1][1], local[1][2], local[1][3]),
				emath::catmull_rom(tx, local[2][0], local[2][1], local[2][2], local[2][3]),
				emath::catmull_rom(tx, local[3][0], local[3][1], local[3][2], local[3][3])
			);
		}
	}

	return large;
}

/// Returns a bunch of line segments as x0, y0, x1, y1.
std::vector<float> iso_surface(int width, int height, const float* values, float iso)
{
	std::vector<float> recentered;
	recentered.reserve(width * height);
	for (int i = 0; i < width * height; ++i) {
		recentered.push_back(values[i] - iso);
	}
	return emilib::marching_squares(width, height, recentered.data());
}

void set_texture(gl::Texture* tex, const ImageRGBA& image)
{
	const auto image_size = gl::Size{image.width(), image.height()};
	tex->set_data(image.data(), image_size, gl::ImageFormat::RGBA32);
}

void save_tga(const char* path, const ImageRGBA& image)
{
	const bool alpha = false;
	CHECK_F(emilib::write_tga(path, image.width(), image.height(), image.data(), alpha));
}

struct FieldGui
{
	Options     options;
	Result      result;
	ImageRGBA   in_values_image;
	ImageRGBA   in_weights_image;
	ImageRGBA   field_image;
	gl::Texture in_values_texture{ "values",  gl::TexParams::clamped_linear()};
	gl::Texture in_weights_texture{"weights", gl::TexParams::clamped_linear()};
	gl::Texture field_texture{     "field",   gl::TexParams::clamped_linear()};
	bool        draw_cells       = true;
	int         field_upsampling = 4;

	FieldGui()
	{
		if (fs::file_exists("bipolar_2d.json")) {
			const auto config = configuru::parse_file("bipolar_2d.json", configuru::JSON);
			configuru::deserialize(&options, config, log_errors);
		}
		calc();
	}

	void save()
	{
		const auto config = configuru::serialize(options);
		configuru::dump_file("bipolar_2d.json", config, configuru::JSON);
	}

	void calc()
	{
		result = generate(options, result.field);

		in_values_image = as_image(options.in_values);
		in_weights_image = as_image(options.in_weights);

		const auto highres_field = wrapping_bicubic_upsample(result.field, field_upsampling);
		field_image = as_image(highres_field);

		set_texture(&in_values_texture, in_values_image);
		set_texture(&in_weights_texture, in_weights_image);
		set_texture(&field_texture, field_image);
	}

	void show_input()
	{
		if (ImGui::Button("Reset all")) {
			result = {};
			options = {};
			calc();
			save();
		}

		if (show_options(&options)) { save(); }
	}

	void paint_iso_line(const ImVec2& canvas_pos, const ImVec2& canvas_size, float iso_value, ImColor color) const
	{
		const Matrixf& source = result.field;
		auto iso_lines = iso_surface(source.width(), source.height(), source.data(), iso_value);
		paint_outline(source.width(), source.height(), iso_lines, canvas_pos, canvas_size, color, false);
	}

	void paint_iso_lines(const ImVec2& canvas_pos, const ImVec2& canvas_size) const
	{
		const auto full_color = ImColor(1.0f, 1.0f, 1.0f, 1.0f);
		const auto half_color = ImColor(5.0f, 5.0f, 5.0f, 1.0f);
		paint_iso_line(canvas_pos, canvas_size, -1.0f, full_color);
		paint_iso_line(canvas_pos, canvas_size, +1.0f, full_color);
		paint_iso_line(canvas_pos, canvas_size, -2.0f, half_color);
		paint_iso_line(canvas_pos, canvas_size, +2.0f, half_color);
	}

	void paint_towers(const AABB2f& out_rect) const
	{
		ImDrawList* draw_list = ImGui::GetWindowDrawList();

		const auto width = options.in_values.width();
		const auto height = options.in_values.height();
		for (const int y : irange(height)) {
			for (const int x : irange(width)) {
				const float v = options.in_values(x, y);
				const float w = options.in_weights(x, y);
				if (v != 0.0 && w > 0.0f) {
					const auto center = remap_pos(Vec2f(x, y), Vec2i(width, height), out_rect);
					const auto r = length(out_rect.size()) / std::hypot(width, height);

					const bool active = std::fabs(result.field(x, y)) <= std::fabs(v);

					const ImColor fill_color = as_color(active ? v : sign(v));
					const ImColor line_color = active ? ImColor(0.8f, 0.8f, 0.8f, 1.0f) : ImColor(0.0f, 0.0f, 0.0f, 1.0f);
					draw_list->AddCircleFilled(center, r, fill_color);
					draw_list->AddCircle(center, r, line_color);
				}
			}
		}
	}

	void show_result()
	{
		const float field_min = *std::min_element(result.field.begin(), result.field.end());
		const float field_max = *std::max_element(result.field.begin(), result.field.end());

		const auto w = result.field.width();
		const auto h = result.field.height();
		ImGui::Text("%d unknowns (%d x %d)", w * h, w, h);
		ImGui::Text("%lu equations", result.eq.rhs.size());
		ImGui::Text("%lu non-zero values in matrix", result.eq.triplets.size());
		ImGui::Text("Log:\n%s", result.log.c_str());

		ImGui::Checkbox("Input cells", &draw_cells);
		ImGui::SameLine();
		ImGui::SliderInt("field_upsampling", &field_upsampling, 1, 10);
		ImGui::Separator();

		in_values_texture.set_params(field_texture.params());
		in_weights_texture.set_params(field_texture.params());
		field_texture.bind(); in_values_texture.bind(); in_weights_texture.bind(); // HACK to apply the params

		const ImVec2 available = ImGui::GetContentRegionAvail();
		const float canvas_width = std::floor(std::min(available.x, (available.y - 128)));
		const float canvas_height = canvas_width * h / w;
		const ImVec2 canvas_size{canvas_width, canvas_height};

		{
			ImGui::Text("field_texture:");
			const ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
			const auto out_rect = AABB2f::from_min_size(canvas_pos, canvas_size);
			ImGui::Image(reinterpret_cast<ImTextureID>(field_texture.id()), canvas_size);
			paint_iso_lines(canvas_pos, canvas_size);
			paint_towers(out_rect);
			ImGui::SetCursorScreenPos(canvas_pos);
			ImGui::InvisibleButton("field_texture_button", canvas_size);
			if (ImGui::IsItemHovered()) {
				const auto pos = ImGui::GetMousePos();
				const int xi = std::round(remap_clamp(pos.x, canvas_pos.x, canvas_pos.x + canvas_size.x, 0.5f, w - 0.5f));
				const int yi = std::round(remap_clamp(pos.y, canvas_pos.y, canvas_pos.y + canvas_size.y, 0.5f, h - 0.5f));

				if (options.in_values.contains_coord(xi, yi)) {
					if (ImGui::IsMouseDown(0)) {
						options.in_values(xi, yi) = +2.0f;
						options.in_weights(xi, yi) = 1.0f;
					} else if (ImGui::IsMouseDown(1)) {
						options.in_values(xi, yi) = -2.0f;
						options.in_weights(xi, yi) = 1.0f;
					} else if (ImGui::IsMouseDown(2)) {
						options.in_values(xi, yi) = 0.0f;
						options.in_weights(xi, yi) = 0.0f;
					}
				}
			}
		}

		ImGui::Separator();
		ImGui::Text("Field min: %f, max: %f", field_min, field_max);

		show_texture_options(&field_texture);
		ImGui::SameLine();
		if (ImGui::Button("Save images")) {
			save_tga("in_values.tga",  in_values_image);
			save_tga("in_weights.tga", in_weights_image);
			save_tga("out_field.tga",  field_image);
		}
	}
};

void show_sdf_fields_for(FieldGui* field_gui)
{
	field_gui->calc();

	ImGui::BeginChild("Input", ImVec2(ImGui::GetWindowContentRegionWidth() * 0.35f, 0), true);
	field_gui->show_input();
	ImGui::EndChild();

	ImGui::SameLine();

	ImGui::BeginChild("Output", ImVec2(ImGui::GetWindowContentRegionWidth() * 0.65f, 0), true);
	field_gui->show_result();
	ImGui::EndChild();
}

} // namespace bipolar_2d

void show_bipolar_2d()
{
	static bipolar_2d::FieldGui s_field_gui;
	bipolar_2d::show_sdf_fields_for(&s_field_gui);
}
