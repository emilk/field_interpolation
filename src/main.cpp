#include <emath/math.hpp>
#include <emilib/gl_lib.hpp>
#include <emilib/gl_lib_opengl.hpp>
#include <emilib/gl_lib_sdl.hpp>
#include <emilib/imgui_gl_lib.hpp>
#include <emilib/imgui_sdl.hpp>
#include <loguru.hpp>

#include "field_1d.hpp"
#include "gui.hpp"
#include "interpolate_2d.hpp"
#include "line_2d.hpp"
#include "sdf_field.hpp"
#include "sine_denoise_1d.hpp"

int main(int argc, char* argv[])
{
	loguru::g_colorlogtostderr = false;
	loguru::init(argc, argv);
	emilib::sdl::Params sdl_params;
	sdl_params.window_name = "2D SDF generator";
	sdl_params.width_points = 1800;
	sdl_params.height_points = 1200;
	auto sdl = emilib::sdl::init(sdl_params);
	emilib::ImGui_SDL imgui_sdl(sdl.width_points, sdl.height_points, sdl.pixels_per_point);
	emilib::gl::bind_imgui_painting();

	bool quit = false;
	while (!quit) {
		SDL_Event event;
		while (SDL_PollEvent(&event)) {
			if (event.type == SDL_QUIT) { quit = true; }
			imgui_sdl.on_event(event);
		}
		emilib::gl::TempViewPort::set_back_buffer_size(
			emath::round_to_int(imgui_sdl.width_pixels()),
			emath::round_to_int(imgui_sdl.height_pixels()));
		imgui_sdl.new_frame();

		ImGui::ShowTestWindow();

		if (ImGui::Begin("1D field interpolation")) {
			show_1d_field_window();
		}
		ImGui::End();

		if (ImGui::Begin("1D sine denoiser")) {
			show_1d_denoiser_window();
		}
		ImGui::End();

		if (ImGui::Begin("2D field interpolation")) {
			show_2d_field_window();
		}
		ImGui::End();

		if (ImGui::Begin("2D SDF")) {
			show_sdf_fields();
		}
		ImGui::End();

		if (ImGui::Begin("line 2d")) {
			show_line_2d();
		}
		ImGui::End();

		glClearColor(0.1f, 0.1f, 0.1f, 0);
		glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

		imgui_sdl.paint();
		emilib::gl::paint_imgui();

		SDL_GL_SwapWindow(sdl.window);
	}
}
