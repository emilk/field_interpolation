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

const float TOWER_HEIGHT = 4.0f;
const float TOWER_WEIGHT = 1.0f;
const float FIRST_ISO = 1.0f;
const float SECOND_ISO = 3.0f;

struct Options
{
	// Input equation: the cell at position X pulls towards in_values[X] with the force in_weights[X]
	Matrixf in_values;
	Matrixf in_weights;

	// Prior for C1 smoothness constraint, i.e.  f[x-1] - 2 * f[x] + f[x+1] = 0
	float   smoothness              =   1.0f;

	// How strongly each cell pulls towards zero (for X where in_weights[X] == 0).
	float   regularization          =   1e-2f;

	// If false, constraints only act to repel from zero
	bool    bilateral               = false;

	// Torus world?
	bool    wrapping                = true;

	int     max_mip_levels          =   1;
	float   mip_connection_strength =   1.0f;
	bool    cg                      = false;
	int     solve_iterations        = 100;
	float   jacobi_weight           =   0.5f;

	Options()
	{
		int width = 128;
		int height = 128;
		in_values = Matrixf(width, height, 0.0);
		in_weights = Matrixf(width, height, 0.0);

		// Set two values:
		in_values(width  * 1 / 3, height / 2) = +TOWER_HEIGHT;
		in_weights(width * 1 / 3, height / 2) =  TOWER_WEIGHT;
		in_values(width  * 2 / 3, height / 2) = -TOWER_HEIGHT;
		in_weights(width * 2 / 3, height / 2) =  TOWER_WEIGHT;
	}
};

} // namespace bipolar_2d

VISITABLE_STRUCT(bipolar_2d::Options, in_values, in_weights, smoothness, regularization, bilateral, wrapping, max_mip_levels, cg, solve_iterations, jacobi_weight);

namespace bipolar_2d {

struct MipIndex
{
	/// Size of each layer (0 = base = largest)
	std::vector<Vec2i> sizes;
	/// Index offset of each layer (sum of the areas of previous layers)
	std::vector<int> offsets;

	MipIndex() {}

	/// max_levels: including base level. Cannot be zero!
	MipIndex(Vec2i size, int max_levels)
	{
		CHECK_GT_F(max_levels, 0);

		int offset = 0;

		sizes.push_back(size);
		offsets.push_back(offset);

		for (int l = 1; l < max_levels; ++l) {
			offset += size.x * size.y;
			size = (size + Vec2i(1, 1)) / 2;
			if (size.x > 1 || size.y > 1) {
				sizes.push_back(size);
				offsets.push_back(offset);
			} else {
				break;
			}
		}

		CHECK_LE_F(num_levels(), max_levels);
	}

	Vec2i base_size() const
	{
		return sizes[0];
	}

	int num_levels() const
	{
		return sizes.size();
	}

	int num_values() const
	{
		return offsets.back() + sizes.back().x * sizes.back().y;
	}

	/// We index the values row-wise, starting at level 0
	int index(int level, Vec2i pos) const
	{
		CHECK_F(0 <= level && level < num_levels());
		CHECK_F(0 <= pos.x && pos.x < sizes[level].x);
		CHECK_F(0 <= pos.y && pos.y < sizes[level].y);
		return offsets[level] + pos.y * sizes[level].x + pos.x;
	}
};

std::vector<float> generate_mip_sum(const MipIndex& index, const Matrixf& base_weights)
{
	std::vector<float> mip;
	mip.reserve(index.num_values());
	mip = base_weights.as_vec();
	mip.resize(index.num_values(), 0.0f);

	for (const int level : irange(index.num_levels() - 1)) {
		for (const int y : irange(index.sizes[level].y)) {
			for (const int x : irange(index.sizes[level].x)) {
				const int this_index = index.index(level, Vec2i(x, y));
				const int above_index = index.index(level + 1, Vec2i(x / 2, y / 2));
				mip[above_index] += mip[this_index];
			}
		}
	}

	return mip;
}

struct Result
{
	fi::LinearEquation eq;
	MipIndex           mip_index;
	std::vector<float> solution; ///< All mip levels.
	Matrixf            field;    ///< Lowest mip level
	std::string        log;      ///< Log captured during solving
};

void log_errors(const std::string& msg)
{
	LOG_F(ERROR, "%s", msg.c_str());
}

void add_tower_weights(fi::LinearEquation* eq, const Options& options, const float* last_field)
{
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
			|| std::fabs(last_field[i]) <= std::fabs(v);

		if (add_constraint) {
			fi::add_equation(eq, fi::Weight{w}, fi::Rhs{v}, {{i, 1.0f}});
		}
	}
}

void add_smoothness(fi::LinearEquation* eq, int offset, Vec2i size, float weight, bool wrapping)
{
	const auto coord = [=](int x, int y) {
		x = (x + size.x) % size.x;
		y = (y + size.y) % size.y;
		return offset + y * size.x + x;
	};

	for (const int y : irange(size.y)) {
		for (const int x : irange(size.x)) {
			if (size.x >= 3 && (wrapping || (0 < x && x < size.x - 1))) {
				add_equation(eq, fi::Weight{weight}, fi::Rhs{0.0f}, {
					{coord(x-1, y), +1.0f},
					{coord(x,   y), -2.0f},
					{coord(x+1, y), +1.0f},
				});
			}

			if (size.y >= 3 && (wrapping || (0 < y && y < size.y - 1))) {
				add_equation(eq, fi::Weight{weight}, fi::Rhs{0.0f}, {
					{coord(x, y-1), +1.0f},
					{coord(x, y),   -2.0f},
					{coord(x, y+1), +1.0f},
				});
			}
		}
	}
}

// Add connections between mip levels
void add_mip_connections(fi::LinearEquation* eq, const Options& options, const MipIndex& mip_index, const std::vector<float>& weight_mip)
{
	for (const auto l : irange(1, mip_index.num_levels())) {
		for (const int y : irange(mip_index.sizes[l].y)) {
			for (const int x : irange(mip_index.sizes[l].x)) {
				const int row = eq->rhs.size();

				// For example:
				//  (L:0, x:4, y:6) + (L:0, x:5, y:6) +
				//  (L:0, x:4, y:7) + (L:0, x:5, y:7)
				//                  =
				//       (L:1, x:2, y:3) * 4

				const Vec2i below_size = mip_index.sizes[l - 1];
				float sum_weight = 0.0;

				for (const int dy : irange(2)) {
					for (const int dx : irange(2)) {
						const auto below = Vec2i(2 * x + dx, 2 * y + dy);
						if (below.x < below_size.x && below.y < below_size.y) {
							const int below_index = mip_index.index(l - 1, below);
							const float weight = weight_mip[below_index] * options.mip_connection_strength;
							if (weight > 0.0f) {
								eq->triplets.emplace_back(row, below_index, weight);
								sum_weight += weight;
							}
						}
					}
				}

				if (sum_weight > 0.0f) {
					const int above_index = mip_index.index(l, Vec2i(x, y));
					eq->triplets.emplace_back(row, above_index, -sum_weight);
					eq->rhs.emplace_back(0.0f);
				}
			}
		}
	}
}

fi::LinearEquation generate_equation(const Options& options, const MipIndex& mip_index, const float* last_field)
{
	LOG_SCOPE_F(1, "generate_equation");

	CHECK_EQ_F(options.in_values.width(), options.in_weights.width());
	CHECK_EQ_F(options.in_values.height(), options.in_weights.height());

	auto size = Vec2i(options.in_values.width(), options.in_values.height());

	Matrixf base_weights = options.in_weights;
	for (float& w : base_weights) {
		if (w <= 0.0f) {
			w = options.regularization;
		}
	}
	const auto weights_mip = generate_mip_sum(mip_index, base_weights);

	fi::LinearEquation eq;
	add_tower_weights(&eq, options, last_field);
	// TODO: add self-sustaining weights (blue area wants to stay blue) ?

	// Add smoothness for each level:
	for (const auto l : irange(0, mip_index.num_levels())) {
		const int scale_factor = (1 << l);
		// TODO: how do we factor in scale_factor?
		// On one hand: higher up = less smooth (because the smoothness is local)
		// On the other hand: higher up there are less constraints, so they need to be stronger.
		const float smoothness = options.smoothness / scale_factor;
		add_smoothness(&eq, mip_index.offsets[l], mip_index.sizes[l], smoothness, options.wrapping);
	}

	add_mip_connections(&eq, options, mip_index, weights_mip);

	return eq;
}

void ram_logger(void* user_data, const loguru::Message& msg)
{
	std::string* log = reinterpret_cast<std::string*>(user_data);
	*log += emilib::strprintf("%s%s%s\n", msg.indentation, msg.prefix, msg.message);
}

Result generate(const Options& options, std::vector<float> last_solution)
{
	const bool first_time = last_solution.empty();

	const auto base_size = Vec2i(options.in_values.width(), options.in_values.height());

	Result result;
	loguru::add_callback("ram_logger", ram_logger, &result.log, loguru::Verbosity_MAX);

	result.mip_index = MipIndex(base_size, options.max_mip_levels);
	last_solution.resize(result.mip_index.num_values(), 0.0f);
	result.eq = generate_equation(options, result.mip_index, last_solution.data());

	if (first_time) {
		result.solution = fi::solve_sparse_linear_exact(result.eq, last_solution.size());
		// result.solution = fi::solve_sparse_linear_fast(result.eq, last_solution.size());
	} else {
		if (options.cg) {
			result.solution = fi::solve_sparse_linear_with_guess(result.eq, last_solution, options.solve_iterations, 0.0f);
		} else {
			result.solution = fi::jacobi_iterations(result.eq, last_solution, options.solve_iterations, options.jacobi_weight);
		}
	}

	if (result.solution.empty()) {
		LOG_F(ERROR, "Failed to solve");
		result.solution = last_solution;
	}

	result.field = Matrixf(base_size.x, base_size.y, result.solution.data());

	loguru::remove_callback("ram_logger");
	return result;
}

RGBA as_color(float field)
{
	if (std::fabs(field) < FIRST_ISO) {
		return RGBA{32, 32, 32, 255};
	};
	if (field > 0.0f) {
		if (field >= SECOND_ISO) {
			return RGBA{255, 0, 0, 255};
		} else {
			return RGBA{128, 0, 0, 255};
		}
	} else {
		if (std::fabs(field) >= SECOND_ISO) {
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

ImageRGBA mip_image(const Result& result)
{
	ImageRGBA image;
	for (const int level : irange(result.mip_index.num_levels())) {
		const Vec2i level_size = result.mip_index.sizes[level];
		const Vec2i img_offset = Vec2i(image.width(), image.height());
		image.resize(img_offset.x + level_size.x, img_offset.y + level_size.y, RGBA{0,0,0,0});

		for (const int y : irange(level_size.y)) {
			for (const int x : irange(level_size.x)) {
				const auto index = result.mip_index.index(level, Vec2i(x, y));
				const auto value = result.solution[index];
				image(img_offset.x + x, img_offset.y + y) = as_color(value);
			}
		}
	}
	return image;
}

bool show_options(Options* options)
{
	bool changed = false;

	int w = options->in_values.width();
	int h = options->in_values.height();

	changed |= ImGui::SliderInt("Width", &w, 16, 256);
	// ImGui::SameLine();
	changed |= ImGui::SliderInt("Height", &h, 16, 256);

	options->in_values.resize(w, h);
	options->in_weights.resize(w, h);

	changed |= ImGui::SliderFloat("smoothness",     &options->smoothness,     0.0f, 10.0f, "%.3f", 4);
	changed |= ImGui::SliderFloat("regularization", &options->regularization, 0.0f,  1.0f, "%.3f", 4);
	changed |= ImGui::Checkbox("bilateral", &options->bilateral);
	changed |= ImGui::Checkbox("wrapping", &options->wrapping);

	changed |= ImGui::SliderInt("max_mip_levels", &options->max_mip_levels, 1, 10);
	if (options->max_mip_levels > 1) {
		changed |= ImGui::SliderFloat("mip_connection_strength", &options->mip_connection_strength, 0.0f,  10.0f, "%.3f", 4);
	}
	changed |= ImGui::Checkbox("CG solver", &options->cg);
	changed |= ImGui::SliderInt("solve_iterations", &options->solve_iterations, 1, 500);
	if (!options->cg) {
		changed |= ImGui::SliderFloat("Jacobi weight", &options->jacobi_weight, 0.0f, 1.0f);
	}

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
	Matrixf     highres_field;
	gl::Texture field_texture{ "field", gl::TexParams::clamped_linear()};
	gl::Texture mip_texture{   "mip",   gl::TexParams::clamped_linear()};
	bool        show_towers      = false;
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
		result = generate(options, result.solution);
		highres_field = wrapping_bicubic_upsample(result.field, field_upsampling);

		set_texture(&field_texture, as_image(highres_field));
		set_texture(&mip_texture, mip_image(result));
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
		const Matrixf& source = highres_field;
		auto iso_lines = iso_surface(source.width(), source.height(), source.data(), iso_value);
		paint_outline(source.width(), source.height(), iso_lines, canvas_pos, canvas_size, color, false);
	}

	void paint_iso_lines(const ImVec2& canvas_pos, const ImVec2& canvas_size) const
	{
		const auto full_color = ImColor(1.0f, 1.0f, 1.0f, 1.0f);
		const auto half_color = ImColor(5.0f, 5.0f, 5.0f, 1.0f);
		paint_iso_line(canvas_pos, canvas_size, -FIRST_ISO, full_color);
		paint_iso_line(canvas_pos, canvas_size, +FIRST_ISO, full_color);
		paint_iso_line(canvas_pos, canvas_size, -SECOND_ISO, half_color);
		paint_iso_line(canvas_pos, canvas_size, +SECOND_ISO, half_color);
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

		ImGui::Checkbox("Towers", &show_towers);
		ImGui::SameLine();
		ImGui::SliderInt("field_upsampling", &field_upsampling, 1, 10);
		ImGui::Separator();

		ImGui::Text("Field min: %f, max: %f", field_min, field_max);

		show_texture_options(&field_texture);
		ImGui::SameLine();

		ImGui::Separator();

		mip_texture.set_params(field_texture.params());
		field_texture.bind(); mip_texture.bind(); // HACK to apply the params

		const ImVec2 available = ImGui::GetContentRegionAvail();
		const auto canvas_size = imgui_helpers::aspect_correct_image_size(
			ImVec2(w, h), available, ImVec2(128, 128));

		static bool s_show_mips = false;
		if (ImGui::RadioButton("Field", s_show_mips == false)) {
			s_show_mips = false;
		}
		ImGui::SameLine();
		if (ImGui::RadioButton("Mip levels", s_show_mips == true)) {
			s_show_mips = true;
		}

		const ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
		const auto out_rect = AABB2f::from_min_size(canvas_pos, canvas_size);

		if (s_show_mips) {
			ImGui::Image(reinterpret_cast<ImTextureID>(mip_texture.id()), canvas_size);
		} else {
			ImGui::Image(reinterpret_cast<ImTextureID>(field_texture.id()), canvas_size);
			paint_iso_lines(canvas_pos, canvas_size);
			if (show_towers) {
				paint_towers(out_rect);
			}
			ImGui::SetCursorScreenPos(canvas_pos);
			ImGui::InvisibleButton("field_texture_button", canvas_size);
			if (ImGui::IsItemHovered()) {
				const auto pos = ImGui::GetMousePos();
				const int xi = std::round(remap_clamp(pos.x, canvas_pos.x, canvas_pos.x + canvas_size.x, 0.5f, w - 0.5f));
				const int yi = std::round(remap_clamp(pos.y, canvas_pos.y, canvas_pos.y + canvas_size.y, 0.5f, h - 0.5f));

				if (options.in_values.contains_coord(xi, yi)) {
					if (ImGui::IsMouseDown(0)) {
						options.in_values(xi, yi) = +TOWER_HEIGHT;
						options.in_weights(xi, yi) = TOWER_WEIGHT;
					} else if (ImGui::IsMouseDown(1)) {
						options.in_values(xi, yi) = -TOWER_HEIGHT;
						options.in_weights(xi, yi) = TOWER_WEIGHT;
					} else if (ImGui::IsMouseDown(2)) {
						options.in_values(xi, yi) = 0.0f;
						options.in_weights(xi, yi) = 0.0f;
					}
				}
			}
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
