#include "interpolate_2d.hpp"

#include <emath/math.hpp>
#include <emilib/gl_lib.hpp>
#include <emilib/imgui_helpers.hpp>

#include <field_interpolation/field_interpolation.hpp>
#include <field_interpolation/sparse_linear.hpp>

#include "gui.hpp"

void show_2d_field_window()
{
	// Based on https://en.wikipedia.org/wiki/Multivariate_interpolation
	float values[4 * 4] = {
		5, 4, 2, 3,
		4, 2, 1, 5,
		6, 3, 5, 2,
		1, 2, 4, 1,
	};

	static int         s_resolution = 64;
	static fi::Weights     s_weights;
	static gl::Texture s_texture{"2d_field", gl::TexParams::clamped_nearest()};

	ImGui::SliderInt("resolution", &s_resolution, 4, 64);
	show_weights(&s_weights);

	fi::LatticeField field{{s_resolution, s_resolution}};
	add_field_constraints(&field, s_weights);

	for (int y = 0; y < 4; ++y) {
		for (int x = 0; x < 4; ++x) {
			const float pos[2] = {
				emath::remap(x, 0.0f, 3.0f, 0.0f, s_resolution - 1.0f),
				emath::remap(y, 0.0f, 3.0f, 0.0f, s_resolution - 1.0f),
			};
			add_value_constraint(&field, pos, values[y * 4 + x], s_weights.data_pos);
			const float zero[2] = {0,0};
			add_gradient_constraint(&field, pos, zero, s_weights.data_gradient, s_weights.gradient_kernel);
		}
	}

	const size_t num_unknowns = s_resolution * s_resolution;
	auto interpolated = solve_sparse_linear_exact(field.eq, num_unknowns);
	if (interpolated.size() != num_unknowns) {
		LOG_F(ERROR, "Failed to find a solution");
		interpolated.resize(num_unknowns, 0.0f);
	}

	const auto heatmap = generate_heatmap(interpolated, 0, 6);
	const auto image_size = gl::Size{static_cast<unsigned>(s_resolution), static_cast<unsigned>(s_resolution)};
	s_texture.set_data(heatmap.data(), image_size, gl::ImageFormat::RGBA32);

	ImVec2 canvas_size{384, 384};
	show_texture_options(&s_texture);
	ImGui::Image(reinterpret_cast<ImTextureID>(s_texture.id()), canvas_size);

	// show_field_equations(field);
}
