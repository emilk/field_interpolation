#pragma once

#include <vector>

#include <visit_struct/visit_struct.hpp>

#include <emilib/gl_lib_fwd.hpp>
#include <emilib/imgui_helpers.hpp>

#include <field_interpolation/field_interpolation.hpp>

namespace fi = field_interpolation;

VISITABLE_STRUCT(fi::Weights, data_pos, data_gradient, model_0, model_1, model_2, model_3, model_4, gradient_smoothness);
VISITABLE_STRUCT(fi::SolveOptions, tile, tile_size, cg, max_iterations, error_tolerance);
VISITABLE_STRUCT(ImVec2, x, y);

using Vec2List = std::vector<ImVec2>;

struct RGBA
{
	uint8_t r, g, b, a;
};

bool show_weights(fi::Weights* weights);

std::vector<RGBA> generate_heatmap(const std::vector<float>& data, float min, float max);

void show_texture_options(emilib::gl::Texture* texture);
