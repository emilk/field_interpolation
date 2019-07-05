#include "sdf_field.hpp"

#include <random>

#include <visit_struct/visit_struct.hpp>

#include <configuru.hpp>
#include <emath/math.hpp>
#include <emilib/dual.hpp>
#include <emilib/file_system.hpp>
#include <emilib/gl_lib.hpp>
#include <emilib/imgui_helpers.hpp>
#include <emilib/irange.hpp>
#include <emilib/marching_squares.hpp>
#include <emilib/tga.hpp>
#include <emilib/timer.hpp>
#include <loguru.hpp>

#include "gui.hpp"

namespace sdf_field {

using namespace emilib;

struct Shape
{
	bool   inverted        = false;
	size_t num_points      = 64;
	float  lopsidedness[2] = {1.0f, 1.0f};

	ImVec2 center          = {0.5f, 0.5f};
	float  radius          =  0.35f;

	float  circleness      =  0;
	size_t polygon_sides   =  3;

	float  rotation        =  0;
};

struct NoiseOptions
{
	int    seed          = 0;
	float  pos_stddev    = 0.005f;
	float  normal_stddev = 0.05f;
	size_t outliers      = 0;
};

struct Options
{
	NoiseOptions       noise;
	size_t             resolution                  = 24;
	std::vector<Shape> shapes;
	fi::Weights        weights;
	float              boundary_weight             =  0.001f;
	bool               exact_solve                 = true;
	int                downscale_factor            =  1;
	bool               cg_from_scratch             = false;
	fi::SolveOptions   solve_options;
	int                marching_squares_upsampling =  1;

	Options()
	{
		shapes.push_back(Shape{});
		{
			Shape c;
			c.inverted = true;
			c.radius = 0.1;
			c.circleness = 1;
			shapes.push_back(c);
		}
	}
};

} // namespace sdf_field

VISITABLE_STRUCT(sdf_field::Shape, inverted, num_points, lopsidedness, center, radius, circleness, polygon_sides, rotation);
VISITABLE_STRUCT(sdf_field::NoiseOptions, seed, pos_stddev, normal_stddev, outliers);
VISITABLE_STRUCT(sdf_field::Options, noise, resolution, shapes, weights, boundary_weight, exact_solve, downscale_factor, cg_from_scratch, solve_options, marching_squares_upsampling);

namespace sdf_field {

struct Result
{
	Vec2List           point_positions;
	Vec2List           point_normals;
	fi::LatticeField   field;
	std::vector<float> sdf;
	std::vector<float> heatmap;
	std::vector<RGBA>  sdf_image;
	std::vector<RGBA>  blob_image;
	std::vector<RGBA>  heatmap_image;
	float              blob_area;
	double             duration_seconds;
};

using Dualf = emilib::Dual<float>;

void log_errors(const std::string& msg)
{
	LOG_F(ERROR, "%s", msg.c_str());
}

auto circle_point(const Shape& shape, Dualf t) -> std::pair<Dualf, Dualf>
{
	Dualf angle = t * emath::TAUf + shape.rotation;
	return std::make_pair(std::cos(angle), std::sin(angle));
}

auto poly_point(const Shape& shape, Dualf t) -> std::pair<Dualf, Dualf>
{
	CHECK_GE_F(shape.polygon_sides, 3u);

	auto polygon_corner = [&](int corner) {
		float angle = emath::TAUf * corner / shape.polygon_sides;
		angle += shape.rotation;
		return ImVec2(std::cos(angle), std::sin(angle));
	};

	int corner_0 = t.real * shape.polygon_sides;

	auto v0 = polygon_corner(corner_0);
	auto v1 = polygon_corner(corner_0 + 1);

	Dualf side_t = t * float(shape.polygon_sides) - float(corner_0);

	Dualf x = v0.x + side_t * (v1.x - v0.x);
	Dualf y = v0.y + side_t * (v1.y - v0.y);
	return std::make_pair(x, y);
}

/// t = [0, 1] along perimeter
auto shape_point(const Shape& shape, Dualf t)
{
	Dualf circle_x, circle_y;
	Dualf poly_x,   poly_y;
	std::tie(circle_x, circle_y) = circle_point(shape, t);
	std::tie(poly_x,   poly_y)   = poly_point(shape,   t);

	Dualf x = emath::lerp(poly_x, circle_x, shape.circleness);
	Dualf y = emath::lerp(poly_y, circle_y, shape.circleness);

	return std::make_pair(ImVec2(x.real, y.real), ImVec2(y.eps, -x.eps));
}

void generate_points(
	Vec2List*    out_positions,
	Vec2List*    out_normals,
	const Shape& shape,
	size_t       min_points)
{
	CHECK_NOTNULL_F(out_positions);
	size_t num_points = std::max(shape.num_points, min_points);

	auto add_point_at = [&](float t) {
		ImVec2 pos, normal;
		auto td = shape.inverted ? Dualf(1.0f - t, -1.0f) : Dualf(t, 1.0f);
		std::tie(pos, normal) = shape_point(shape, td);

		pos.x = shape.center.x + shape.radius * pos.x;
		pos.y = shape.center.y + shape.radius * pos.y;

		out_positions->emplace_back(pos);
		if (out_normals) {
			out_normals->emplace_back(normal);
		}
	};

	int num_points_in_first_half = std::round(shape.lopsidedness[0] * num_points / 2);
	for (size_t i = 0; i < num_points_in_first_half; ++i) {
		add_point_at(0.5f * float(i) / num_points_in_first_half);
	}

	int num_points_in_second_half = std::round(shape.lopsidedness[1] * num_points / 2);
	for (size_t i = 0; i < num_points_in_second_half; ++i) {
		add_point_at(0.5f + 0.5f * i / num_points_in_second_half);
	}
}

float area(const std::vector<Shape>& shapes)
{
	double expected_area = 0;
	for (const auto& shape : shapes) {
		Vec2List positions;
		generate_points(&positions, nullptr, shape, 2048);

		std::vector<float> line_segments;
		for (const auto i : emilib::indices(positions)) {
			line_segments.push_back(positions[i].x);
			line_segments.push_back(positions[i].y);
			line_segments.push_back(positions[(i + 1) % positions.size()].x);
			line_segments.push_back(positions[(i + 1) % positions.size()].y);
		}
		expected_area += emilib::calc_area(line_segments.size() / 4, line_segments.data());
	}
	return expected_area;
}

Vec2List on_lattice(const Vec2List& positions, float resolution)
{
	Vec2List lattice_positions;

	for (const auto& pos : positions) {
		ImVec2 on_lattice = pos;
		on_lattice.x *= (resolution - 1.0f);
		on_lattice.y *= (resolution - 1.0f);
		lattice_positions.push_back(on_lattice);
	}

	return lattice_positions;
}

fi::LatticeField generate_sdf_field(int width, int height, const Options& options, const Vec2List& positions, const Vec2List& normals)
{
	CHECK_EQ_F(positions.size(), normals.size());
	auto field = sdf_from_points(
			{width, height}, options.weights, positions.size(), &positions[0].x, &normals[0].x, nullptr);

	if (options.boundary_weight > 0)
	{
		const float weight = options.boundary_weight;
		for (int y = 0; y < height; ++y)
		{
			for (int x = 0; x < width; ++x)
			{
				bool is_on_border =
					x == 0 || x == width - 1 ||
					y == 0 || y == height - 1;

				if (is_on_border)
				{
					float closest_dist_sq = std::numeric_limits<float>::infinity();
					for (const auto& pos : positions)
					{
						const float dx = pos.x - x;
						const float dy = pos.y - y;
						closest_dist_sq = std::min(closest_dist_sq, dx * dx + dy * dy);
					}
					const float expected_value = std::sqrt(closest_dist_sq);
					const int index = y * width + x;
					add_equation(&field.eq, fi::Weight{weight}, {expected_value}, {
						{index, 1.0f}
					});
				}
			}
		}
	}

	return field;
}

auto generate_sdf(const Vec2List& positions, const Vec2List& normals, const Options& options,
                  const std::vector<float>& last_solution)
{
	LOG_SCOPE_F(1, "generate_sdf");
	CHECK_EQ_F(positions.size(), normals.size());

	const int width = options.resolution;
	const int height = options.resolution;

	static_assert(sizeof(ImVec2) == 2 * sizeof(float), "Pack");

	const size_t num_unknowns = width * height;

	const Vec2List large_lattice_positions = on_lattice(positions, options.resolution);

	const auto field = generate_sdf_field(width, height, options, large_lattice_positions, normals);

	std::vector<float> sdf;
	if (options.exact_solve) {
		sdf = solve_sparse_linear_exact(field.eq, num_unknowns);
	} else {
		if (2 <= options.downscale_factor) {
			const int resolution_small = (options.resolution + options.downscale_factor - 1) / options.downscale_factor;

			const int num_unknowns_small = resolution_small * resolution_small;
			const std::vector<int> sizes_small{resolution_small, resolution_small};

			const Vec2List small_lattice_positions = on_lattice(positions, resolution_small);

			const auto field_small = generate_sdf_field(resolution_small, resolution_small, options, small_lattice_positions, normals);

			const auto solution_small = solve_sparse_linear_exact(field_small.eq, num_unknowns_small);

			sdf = fi::upscale_field(solution_small.data(), sizes_small, {width, height});

			for (auto& value : sdf) {
				value *= options.downscale_factor;
			}
		} else if (options.cg_from_scratch || last_solution.size() != num_unknowns) {
			sdf = std::vector<float>(num_unknowns, 0.0f);
		} else {
			sdf = last_solution;
		}

		sdf = solve_tiled_with_guess(field.eq, sdf, field.sizes, options.solve_options);
	}

	if (sdf.size() != num_unknowns) {
		LOG_F(ERROR, "Failed to find a solution");
		sdf.resize(num_unknowns, 0.0f);
	}

	return std::make_tuple(field, sdf);
}

void perturb_points(Vec2List* positions, Vec2List* normals, const NoiseOptions& options)
{
	std::default_random_engine rng(options.seed);
	std::normal_distribution<float> pos_noise(0.0, options.pos_stddev);
	std::normal_distribution<float> dir_noise(0.0, options.normal_stddev);

	for (auto& pos : *positions) {
		pos.x += pos_noise(rng);
		pos.y += pos_noise(rng);
	}
	for (auto& normal : *normals) {
		float angle = std::atan2(normal.y, normal.x);
		angle += dir_noise(rng);
		normal.x = std::cos(angle);
		normal.y = std::sin(angle);
	}

	std::uniform_real_distribution<float> random_pos(0.0f, 1.0f);
	std::normal_distribution<float> random_normal;
	for (size_t i = 0; i < options.outliers; ++i) {
		positions->emplace_back(random_pos(rng), random_pos(rng));
		normals->emplace_back(random_normal(rng), random_normal(rng));
	}

	for (auto& normal : *normals) {
		normal /= std::hypot(normal.x, normal.y);
	}
}

Result generate(const Options& options, const std::vector<float>& last_solution)
{
	ERROR_CONTEXT("resolution", options.resolution);

	emilib::Timer timer;
	const int resolution = options.resolution;

	Result result;

	for (const auto& shape : options.shapes) {
		generate_points(&result.point_positions, &result.point_normals, shape, 0);
	}
	perturb_points(&result.point_positions, &result.point_normals, options.noise);

	std::tie(result.field, result.sdf) = generate_sdf(result.point_positions, result.point_normals, options, last_solution);
	result.heatmap = generate_error_map(result.field.eq.triplets, result.sdf, result.field.eq.rhs);
	result.heatmap_image = generate_heatmap(result.heatmap, 0, *max_element(result.heatmap.begin(), result.heatmap.end()));
	CHECK_EQ_F(result.heatmap_image.size(), resolution * resolution);

	double area_pixels = 0;

	float max_abs_dist = 1e-6f;
	for (const float dist : result.sdf) {
		max_abs_dist = std::max(max_abs_dist, std::abs(dist));
	}

	for (const float dist : result.sdf) {
		const uint8_t dist_u8 = std::min<float>(255, 255 * std::abs(dist) / max_abs_dist);
		const uint8_t inv_dist_u8 = 255 - dist_u8;;
		if (dist < 0) {
			result.sdf_image.emplace_back(RGBA{inv_dist_u8, inv_dist_u8, 255, 255});
		} else {
			result.sdf_image.emplace_back(RGBA{255, inv_dist_u8, inv_dist_u8, 255});
		}

		float insideness = 1 - std::max(0.0, std::min(1.0, (dist + 0.5) * 2));
		const uint8_t color = 255 * insideness;
		result.blob_image.emplace_back(RGBA{color, color, color, 255});

		area_pixels += insideness;
	}

	result.blob_area = area_pixels / emath::sqr(resolution - 1);

	result.duration_seconds = timer.secs();
	return result;
}

bool show_shape_options(Shape* shape)
{
	bool changed = false;

	ImGui::Text("Shape:");
	changed |= ImGui::Checkbox("inverted (hole)",   &shape->inverted);
	changed |= ImGuiPP::SliderSize("num_points",    &shape->num_points,    1, 100000, 4);
	changed |= ImGui::SliderFloat2("lopsidedness",  shape->lopsidedness,   0,    2);
	changed |= ImGui::SliderFloat2("center",        &shape->center.x,      0,    1);
	changed |= ImGui::SliderFloat("radius",         &shape->radius,        0,    1);
	changed |= ImGui::SliderFloat("circleness",     &shape->circleness,   -1,    5);
	changed |= ImGuiPP::SliderSize("polygon_sides", &shape->polygon_sides, 3,    8);
	changed |= ImGui::SliderAngle("rotation",       &shape->rotation,      0,  360);
	return changed;
}

bool show_noise_options(NoiseOptions* options)
{
	bool changed = false;
	ImGui::PushItemWidth(ImGui::GetWindowContentRegionWidth() * 0.3f);
	ImGui::Text("Noise:");
	changed |= ImGui::SliderInt("seed      ", &options->seed, 0, 100);
	ImGui::SameLine();
	changed |= ImGuiPP::SliderSize("outliers", &options->outliers, 0, 50);
	changed |= ImGui::SliderFloat("pos_stddev", &options->pos_stddev, 0,   0.1, "%.4f");
	ImGui::SameLine();
	changed |= ImGui::SliderAngle("normal_stddev", &options->normal_stddev, 0, 360);
	ImGui::PopItemWidth();
	return changed;
}

bool show_solve_options(fi::SolveOptions* options)
{
	bool changed = false;
	changed |= ImGui::Checkbox("tile", &options->tile);
	if (options->tile) {
		changed |= ImGui::SliderInt("tile_size", &options->tile_size, 0, 200);
	}
	changed |= ImGui::Checkbox("cg", &options->cg);
	if (options->cg) {
		changed |= ImGui::SliderInt("max_iterations", &options->max_iterations, 0, 200);
		changed |= ImGui::SliderFloat("error_tolerance", &options->error_tolerance, 1e-6f, 1, "%.6f", 4);
	}
	return changed;
}

bool show_options(Options* options)
{
	bool changed = false;

	if (ImGui::Button("Reset all")) {
		*options = {};
		changed = true;
	}
	changed |= ImGuiPP::SliderSize("resolution", &options->resolution, 4, 256);
	ImGui::Separator();
	for (const int i : emilib::indices(options->shapes)) {
		ImGui::PushID(i);
		changed |= show_shape_options(&options->shapes[i]);
		ImGui::PopID();
		ImGui::Separator();
	}
	if (options->shapes.size() >= 2 && ImGui::Button("Remove shape")) {
		options->shapes.pop_back();
		changed = true;
		ImGui::SameLine();
	}
	if (ImGui::Button("Add shape")) {
		options->shapes.push_back(Shape{});
		changed = true;
	}
	ImGui::Separator();
	changed |= show_noise_options(&options->noise);
	ImGui::Separator();
	changed |= show_weights(&options->weights);
	changed |= ImGui::SliderFloat("boundary_weight", &options->boundary_weight, 0, 1000, "%.3f", 4);

	changed |= ImGui::Checkbox("Exact solve", &options->exact_solve);
	if (!options->exact_solve) {
		changed |= ImGui::SliderInt("downscale_factor", &options->downscale_factor, 1, 10);
		changed |= ImGui::Checkbox("From scratch", &options->cg_from_scratch);
		if (!options->cg_from_scratch) {
			ImGui::SameLine();
			changed |= ImGui::Button("calc");
		}
		changed |= show_solve_options(&options->solve_options);
	}

	changed |= ImGui::SliderInt("marching_squares_upsampling", &options->marching_squares_upsampling, 1, 10);

	return changed;
}

void show_cells(const Options& options, ImVec2 canvas_pos, ImVec2 canvas_size)
{
	ImDrawList* draw_list = ImGui::GetWindowDrawList();

	if (options.resolution < 64) {
		// Draw sample points
		for (size_t xi : emilib::irange<size_t>(0, options.resolution)) {
			for (size_t yi : emilib::irange<size_t>(0, options.resolution)) {
				const float x = static_cast<float>(xi) / (options.resolution - 1.0f);
				const float y = static_cast<float>(yi) / (options.resolution - 1.0f);
				const float center_x = canvas_pos.x + canvas_size.x * x;
				const float center_y = canvas_pos.y + canvas_size.y * y;
				draw_list->AddCircleFilled({center_x, center_y}, 1, ImColor(1.0f, 1.0f, 1.0f, 0.25f), 4);
			}
		}
	}
}

void show_points(const Options& options, const Vec2List& positions, const Vec2List& normals,
	ImVec2 canvas_pos, ImVec2 canvas_size)
{
	CHECK_EQ_F(positions.size(), normals.size());
	if (positions.size() > 2000) { return; }

	ImDrawList* draw_list = ImGui::GetWindowDrawList();

	for (const auto pi : emilib::indices(positions)) {
		ImVec2 center;
		center.x = canvas_pos.x + canvas_size.x * positions[pi].x;
		center.y = canvas_pos.y + canvas_size.y * positions[pi].y;
		draw_list->AddCircleFilled(center, 1, ImColor(1.0f, 1.0f, 1.0f, 1.0f), 4);

		if (positions.size() < 1000) {
			const float arrow_len = 5;
			draw_list->AddLine(center, ImVec2{center.x + arrow_len * normals[pi].x, center.y + arrow_len * normals[pi].y}, ImColor(1.0f, 1.0f, 1.0f, 0.75f));
		}
	}
}

void show_outline(
	size_t                    resolution,
	const std::vector<float>& lines,
	ImVec2                    canvas_pos,
	ImVec2                    canvas_size,
	ImColor                   color,
	bool                      draw_blob_normals)
{
	ImDrawList* draw_list = ImGui::GetWindowDrawList();

	CHECK_F(lines.size() % 4 == 0);

	for (size_t i = 0; i < lines.size(); i += 4) {
		float x0 = lines[i + 0];
		float y0 = lines[i + 1];
		float x1 = lines[i + 2];
		float y1 = lines[i + 3];

		x0 = canvas_pos.x + canvas_size.x * (x0 / (resolution - 1.0f));
		y0 = canvas_pos.y + canvas_size.y * (y0 / (resolution - 1.0f));
		x1 = canvas_pos.x + canvas_size.x * (x1 / (resolution - 1.0f));
		y1 = canvas_pos.y + canvas_size.y * (y1 / (resolution - 1.0f));

		draw_list->AddLine({x0, y0}, {x1, y1}, color);

		if (draw_blob_normals) {
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

std::vector<float> bicubic_upsample(int* io_width, int* io_height, const float* values, int upsample)
{
	CHECK_GT_F(upsample, 1);

	const int small_width = *io_width;
	const int small_height = *io_height;
	const size_t large_width = upsample * small_width - upsample + 1;
	const size_t large_height = upsample * small_height - upsample + 1;

	const auto value_at = [&](int x, int y)
	{
		x = emath::clamp(x, 0, small_width - 1);
		y = emath::clamp(y, 0, small_height - 1);
		return values[y * small_width + x];
	};

	std::vector<float> large;
	large.reserve(large_width * large_height);
	for (int ly = 0; ly < large_height; ++ly) {
		for (int lx = 0; lx < large_width; ++lx) {
			float tx = static_cast<float>(lx % upsample) / static_cast<float>(upsample);
			float ty = static_cast<float>(ly % upsample) / static_cast<float>(upsample);

			int sx = lx / upsample;
			int sy = ly / upsample;

			float local[4][4] {
				{ value_at(sx - 1, sy - 1), value_at(sx + 0, sy - 1), value_at(sx + 1, sy - 1), value_at(sx + 2, sy - 1) },
				{ value_at(sx - 1, sy + 0), value_at(sx + 0, sy + 0), value_at(sx + 1, sy + 0), value_at(sx + 2, sy + 0) },
				{ value_at(sx - 1, sy + 1), value_at(sx + 0, sy + 1), value_at(sx + 1, sy + 1), value_at(sx + 2, sy + 1) },
				{ value_at(sx - 1, sy + 2), value_at(sx + 0, sy + 2), value_at(sx + 1, sy + 2), value_at(sx + 2, sy + 2) },
			};

			large.push_back(
				emath::catmull_rom(ty,
					emath::catmull_rom(tx, local[0][0], local[0][1], local[0][2], local[0][3]),
					emath::catmull_rom(tx, local[1][0], local[1][1], local[1][2], local[1][3]),
					emath::catmull_rom(tx, local[2][0], local[2][1], local[2][2], local[2][3]),
					emath::catmull_rom(tx, local[3][0], local[3][1], local[3][2], local[3][3])
				)
			);
		}
	}

	*io_width = large_width;
	*io_height = large_height;

	return large;
}

std::vector<float> iso_surface(int width, int height, const float* values, float iso)
{
	std::vector<float> iso_at_zero;
	iso_at_zero.reserve(width * height);
	for (int i = 0; i < width * height; ++i) {
		iso_at_zero.push_back(values[i] - iso);
	}

	return emilib::marching_squares(width, height, iso_at_zero.data());
}

struct FieldGui
{
	Options options;
	Result result;
	gl::Texture sdf_texture{"sdf", gl::TexParams::clamped_nearest()};
	gl::Texture blob_texture{"blob", gl::TexParams::clamped_nearest()};
	gl::Texture heatmap_texture{"heatmap", gl::TexParams::clamped_nearest()};
	bool draw_points = true;
	bool draw_cells = true;
	bool draw_iso_lines = true;
	bool draw_blob_normals = false;
	float iso_spacing = 2;

	FieldGui()
	{
		if (fs::file_exists("sdf_input.json")) {
			const auto config = configuru::parse_file("sdf_input.json", configuru::JSON);
			configuru::deserialize(&options, config, log_errors);
		}
		calc();
	}

	void calc()
	{
		const std::vector<float> last_solution = result.sdf;
		result = generate(options, last_solution);
		const auto image_size = gl::Size{static_cast<int>(options.resolution), static_cast<int>(options.resolution)};
		sdf_texture.set_data(result.sdf_image.data(),         image_size, gl::ImageFormat::RGBA32);
		blob_texture.set_data(result.blob_image.data(),       image_size, gl::ImageFormat::RGBA32);
		heatmap_texture.set_data(result.heatmap_image.data(), image_size, gl::ImageFormat::RGBA32);
	}

	void show_input()
	{
		if (show_options(&options)) {
			calc();
			const auto config = configuru::serialize(options);
			configuru::dump_file("sdf_input.json", config, configuru::JSON);
		}
	}

	void show_result()
	{
		const float iso_min = *min_element(result.sdf.begin(), result.sdf.end());
		const float iso_max = *max_element(result.sdf.begin(), result.sdf.end());

		std::vector<float> iso_source = result.sdf;
		int iso_width = options.resolution;
		int iso_height = options.resolution;
		if (options.marching_squares_upsampling > 1) {
			iso_source = bicubic_upsample(&iso_width, &iso_height, iso_source.data(), options.marching_squares_upsampling);
		}

		std::vector<float> zero_lines = iso_surface(iso_width, iso_height, iso_source.data(), 0.0f);
		const float lines_area = emilib::calc_area(zero_lines.size() / 4, zero_lines.data()) / emath::sqr(iso_width - 1);

		ImGui::Text("%lu unknowns", options.resolution * options.resolution);
		ImGui::Text("%lu equations", result.field.eq.rhs.size());
		ImGui::Text("%lu non-zero values in matrix", result.field.eq.triplets.size());
		ImGui::Text("Calculated in %.3f s", result.duration_seconds);
		ImGui::Text("Model area: %.3f, marching squares area: %.3f, sdf blob area: %.3f",
			area(options.shapes), lines_area, result.blob_area);

		ImGui::Checkbox("Input points", &draw_points);
		ImGui::SameLine();
		ImGui::Checkbox("Input cells", &draw_cells);
		ImGui::SameLine();
		ImGui::Checkbox("Output blob", &draw_iso_lines);
		if (draw_iso_lines) {
			ImGui::SameLine();
			ImGui::Checkbox("Output normals", &draw_blob_normals);
			ImGui::SameLine();
			ImGui::PushItemWidth(128);
			ImGui::SliderFloat("Iso spacing", &iso_spacing, 1, 10, "%.0f");
		}

		ImVec2 available = ImGui::GetContentRegionAvail();
		float image_width = std::floor(std::min(available.x / 2, (available.y - 64) / 2));
		ImVec2 canvas_size{image_width, image_width};
		ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
		ImGui::InvisibleButton("canvas", canvas_size);
		if (draw_cells) { show_cells(options, canvas_pos, canvas_size); }
		if (draw_points) { show_points(options, result.point_positions, result.point_normals, canvas_pos, canvas_size); }
		if (draw_iso_lines) {
			for (int i = emath::floor_to_int(iso_min / iso_spacing); i <= emath::ceil_to_int(iso_max / iso_spacing); ++i) {
				auto iso_lines = iso_surface(iso_width, iso_height, iso_source.data(), i * iso_spacing);
				int color = i == 0 ? ImColor(1.0f, 0.0f, 0.0f, 1.0f) : ImColor(0.5f, 0.5f, 0.5f, 0.5f);
				show_outline(iso_width, iso_lines, canvas_pos, canvas_size, color, i == 0 && draw_blob_normals);
			}
		}

		blob_texture.set_params(sdf_texture.params());
		heatmap_texture.set_params(sdf_texture.params());

		// HACK to apply the params:
		sdf_texture.bind(); blob_texture.bind(); heatmap_texture.bind();

		ImGui::SameLine();
		ImGui::Image(reinterpret_cast<ImTextureID>(heatmap_texture.id()), canvas_size);

		ImGui::Text("Max error: %f", *max_element(result.heatmap.begin(), result.heatmap.end()));

		ImGui::Image(reinterpret_cast<ImTextureID>(sdf_texture.id()), canvas_size);
		ImGui::SameLine();
		ImGui::Image(reinterpret_cast<ImTextureID>(blob_texture.id()), canvas_size);

		ImGui::Text("Field min: %f, max: %f", iso_min, iso_max);

		show_texture_options(&sdf_texture);
		ImGui::SameLine();
		if (ImGui::Button("Save images")) {
			const auto res = options.resolution;
			const bool alpha = false;
			CHECK_F(emilib::write_tga("heatmap.tga", res, res, result.heatmap_image.data(), alpha));
			CHECK_F(emilib::write_tga("sdf.tga",     res, res, result.sdf_image.data(),     alpha));
			CHECK_F(emilib::write_tga("blob.tga",    res, res, result.blob_image.data(),    alpha));
		}
	}
};

void show_sdf_fields_for(FieldGui* field_gui)
{
	ImGui::BeginChild("Input", ImVec2(ImGui::GetWindowContentRegionWidth() * 0.35f, 0), true);
	field_gui->show_input();
	ImGui::EndChild();

	ImGui::SameLine();

	ImGui::BeginChild("Output", ImVec2(ImGui::GetWindowContentRegionWidth() * 0.65f, 0), true);
	field_gui->show_result();
	ImGui::EndChild();
}

} // namespace sdf_field

void show_sdf_fields()
{
	static sdf_field::FieldGui s_field_gui;
	sdf_field::show_sdf_fields_for(&s_field_gui);
}
