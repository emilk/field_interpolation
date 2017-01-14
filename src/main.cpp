#include <vector>
#include <random>

#include <imgui/imgui.h>
#include <SDL2/SDL.h>

#include <emilib/gl_lib.hpp>
#include <emilib/gl_lib_opengl.hpp>
#include <emilib/gl_lib_sdl.hpp>
#include <emilib/imgui_gl_lib.hpp>
#include <emilib/imgui_helpers.hpp>
#include <emilib/imgui_sdl.hpp>
#include <emilib/tga.hpp>
#include <loguru.hpp>

#include "sdf.hpp"

struct RGBA
{
	uint8_t r, g, b, a;
};

struct PointOptions
{
	size_t num_points = 64;

	float  center     =   0.5f;
	float  radius     =   0.35f;

	float  pos_noise  =   0.05f;
	float  dir_noise  =   0.01f;
};

struct Options
{
	size_t       resolution = 100;
	PointOptions points;
	Strengths    strengths;
};

struct Result
{
	std::vector<float> sdf;
	std::vector<RGBA>  points_image;
	std::vector<RGBA>  sdf_image;
	std::vector<RGBA>  blob_image;
	float              blob_area;
};

std::vector<Point> generate_points(const PointOptions& options)
{
	std::default_random_engine rng(0);

	std::normal_distribution<float> pos_noise(0.0, options.pos_noise);
	std::normal_distribution<float> dir_noise(0.0, options.dir_noise);

	std::vector<Point> points;

	for (size_t i = 0; i < options.num_points; ++i)
	{
		float angle = i * M_PI * 2 / options.num_points;
		float x = options.center + options.radius * std::cos(angle);
		float y = options.center + options.radius * std::sin(angle);
		float dx = std::cos(angle);
		float dy = std::sin(angle);
		points.push_back(Point{
			x  + pos_noise(rng), y  + pos_noise(rng),
			dx + dir_noise(rng), dy + dir_noise(rng)});
	}

	return points;
}

Result generate(const Options& options)
{
	const int resolution = options.resolution;

	Result result;
	const auto points = generate_points(options.points);

	result.points_image.resize(resolution * resolution, RGBA{0, 0, 0, 255});

	auto index = [=](int x, int y) { return y * resolution + x; };

	for (const auto& point : points) {
		int x = std::round(point.x * resolution);
		int y = std::round(point.y * resolution);

		if (x < 1 || x >= resolution - 1) { continue; }
		if (y < 1 || y >= resolution - 1) { continue; }

		result.points_image[index(x, y)] = RGBA{255, 255, 255, 255};
		result.points_image[index(std::round(point.x * resolution + point.dx),
		                         std::round(point.y * resolution + point.dy))] = RGBA{127, 127, 127, 255};
	}

	result.sdf = generate_sdf(options.resolution, points, options.strengths);
	CHECK_EQ_F(result.sdf.size(), resolution * resolution);

	double area_pixels = 0;

	const float COLOR_SCALE = 10;

	for (const float dist : result.sdf) {
		const uint8_t dist_u8 = std::min<float>(255, std::abs(dist) * COLOR_SCALE);
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

	result.blob_area = area_pixels / (resolution * resolution);

	return result;
}

bool showPointOptions(PointOptions* options)
{
	bool changed = false;

	ImGui::Text("Points:");
	changed |= ImGuiPP::SliderSize("num_points", &options->num_points, 1, 1024, 2);
	changed |= ImGui::SliderFloat("center",      &options->center,     0,    1);
	changed |= ImGui::SliderFloat("radius",      &options->radius,     0,    1);
	changed |= ImGui::SliderFloat("pos_noise",   &options->pos_noise,  0,    1, "%.4f", 4);
	changed |= ImGui::SliderFloat("dir_noise",   &options->dir_noise,  0,    1, "%.4f", 4);

	return changed;
}

bool showStrengths(Strengths* strengths)
{
	bool changed = false;

	ImGui::Text("How much we trust the data:");
	changed |= ImGui::SliderFloat("data_pos",    &strengths->data_pos,    0, 10, "%.4f", 4);
	changed |= ImGui::SliderFloat("data_normal", &strengths->data_normal, 0, 10, "%.4f", 4);
	ImGui::Text("How much we trust the model:");
	changed |= ImGui::SliderFloat("model_0",     &strengths->model_0,     0, 10, "%.4f", 4);
	changed |= ImGui::SliderFloat("model_1",     &strengths->model_1,     0, 10, "%.4f", 4);
	changed |= ImGui::SliderFloat("model_2",     &strengths->model_2,     0, 10, "%.4f", 4);

	return changed;
}

bool showOptions(Options* options)
{
	bool changed = false;

	if (ImGui::Button("Reset all")) {
		*options = {};
		changed = true;
	}
	changed |= ImGuiPP::SliderSize("resolution", &options->resolution, 4, 256);
	ImGui::Separator();
	changed |=showPointOptions(&options->points);
	ImGui::Separator();
	changed |= showStrengths(&options->strengths);

	return changed;
}

int main(int argc, char* argv[])
{
	loguru::g_colorlogtostderr = false;
	loguru::init(argc, argv);

	emilib::sdl::Params sdl_params;
	sdl_params.window_name = "2D SDF generator";
	auto sdl = emilib::sdl::init(sdl_params);

	emilib::ImGui_SDL imgui_sdl(sdl.width_points, sdl.height_points, sdl.pixels_per_point);

	gl::bind_imgui_painting();

	Options options;
	auto result = generate(options);

	gl::Texture points_texture{"points", gl::TexParams::clamped_nearest()};
	gl::Texture sdf_texture{"sdf", gl::TexParams::clamped_nearest()};
	gl::Texture blob_texture{"blob", gl::TexParams::clamped_nearest()};

	const auto image_size = gl::Size{static_cast<unsigned>(options.resolution), static_cast<unsigned>(options.resolution)};
	points_texture.set_data(result.points_image.data(), image_size, gl::ImageFormat::RGBA32);
	sdf_texture.set_data(result.sdf_image.data(),       image_size, gl::ImageFormat::RGBA32);
	blob_texture.set_data(result.blob_image.data(),     image_size, gl::ImageFormat::RGBA32);

	bool quit = false;
	while (!quit) {
		SDL_Event event;
		while (SDL_PollEvent(&event)) {
			if (event.type == SDL_QUIT) { quit = true; }
			imgui_sdl.on_event(event);
		}
		imgui_sdl.new_frame();

		if (showOptions(&options)) {
			result = generate(options);
			const auto image_size = gl::Size{static_cast<unsigned>(options.resolution), static_cast<unsigned>(options.resolution)};
			points_texture.set_data(result.points_image.data(), image_size, gl::ImageFormat::RGBA32);
			sdf_texture.set_data(result.sdf_image.data(),       image_size, gl::ImageFormat::RGBA32);
			blob_texture.set_data(result.blob_image.data(),     image_size, gl::ImageFormat::RGBA32);
		}

		ImGui::Image(reinterpret_cast<ImTextureID>(points_texture.id()), {256, 256});
		ImGui::SameLine();
		ImGui::Image(reinterpret_cast<ImTextureID>(sdf_texture.id()), {256, 256});
		ImGui::SameLine();
		ImGui::Image(reinterpret_cast<ImTextureID>(blob_texture.id()), {256, 256});

		ImGui::Text("Model area: %.3f, sdf blob area: %.3f",
		    M_PI * std::pow(options.points.radius, 2), result.blob_area);

		if (ImGui::Button("Quit!")) {
			quit = true;
		}

		if (ImGui::Button("Save images")) {
			const auto res = options.resolution;
			const bool alpha = false;
			CHECK_F(emilib::write_tga("points.tga", res, res, result.points_image.data(), alpha));
			CHECK_F(emilib::write_tga("sdf.tga",    res, res, result.sdf_image.data(),    alpha));
			CHECK_F(emilib::write_tga("blob.tga",   res, res, result.blob_image.data(),   alpha));
		}

		glClearColor(0.1f, 0.1f, 0.1f, 0);
		glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

		imgui_sdl.paint();

		SDL_GL_SwapWindow(sdl.window);
	}
}
