#include "gui.hpp"

#include <stb/stb_image.h>

#include <emath/math.hpp>
#include <emilib/gl_lib.hpp>
#include <emilib/imgui_helpers.hpp>
#include <emilib/scope_exit.hpp>
#include <loguru.hpp>

bool show_weights(fi::Weights* weights)
{
	bool changed = false;

	{
		ImGui::PushID("Value kernel");
		ImGui::Text("Value kernel:");
		ImGui::SameLine();
		changed |= ImGuiPP::RadioButtonEnum("nearest-neighbor", &weights->value_kernel, fi::ValueKernel::kNearestNeighbor);
		ImGui::SameLine();
		changed |= ImGuiPP::RadioButtonEnum("n-linear-interpolation", &weights->value_kernel, fi::ValueKernel::kLinearInterpolation);
		ImGui::PopID();
	}

	{
		ImGui::PushID("Gradient kernel");
		ImGui::Text("Gradient kernel:");
		ImGui::SameLine();
		changed |= ImGuiPP::RadioButtonEnum("nearest-neighbor", &weights->gradient_kernel, fi::GradientKernel::kNearestNeighbor);
		ImGui::SameLine();
		changed |= ImGuiPP::RadioButtonEnum("cell edges", &weights->gradient_kernel, fi::GradientKernel::kCellEdges);
		ImGui::SameLine();
		changed |= ImGuiPP::RadioButtonEnum("n-linear-interpolation", &weights->gradient_kernel, fi::GradientKernel::kLinearInterpolation);
		ImGui::PopID();
	}

	if (ImGui::Button("Reset weights")) {
		*weights = {};
		changed = true;
	}
	ImGui::Text("How much we trust the data:");
	changed |= ImGui::SliderFloat("data_pos",      &weights->data_pos,      0, 1000, "%.3f", 4);
	changed |= ImGui::SliderFloat("data_gradient", &weights->data_gradient, 0, 1000, "%.3f", 4);
	ImGui::Text("How much we trust the model:");
	changed |= ImGui::SliderFloat("f(0) = 0 (regularization)",    &weights->model_0, 0, 1000, "%.3f", 4);
	changed |= ImGui::SliderFloat("f'(0) = 0 (flatness)",         &weights->model_1, 0, 1000, "%.3f", 4);
	changed |= ImGui::SliderFloat("f''(0) = 0 (C1 smoothness)",   &weights->model_2, 0, 1000, "%.3f", 4);
	changed |= ImGui::SliderFloat("f'''(0) = 0 (C2 smoothness)",  &weights->model_3, 0, 1000, "%.3f", 4);
	changed |= ImGui::SliderFloat("f''''(0) = 0 (C3 smoothness)", &weights->model_4, 0, 1000, "%.3f", 4);
	changed |= ImGui::SliderFloat("Gradient smoothness)", &weights->gradient_smoothness, 0, 1000, "%.3f", 4);

	return changed;
}

std::vector<RGBA> generate_heatmap(const std::vector<float>& data, float min, float max)
{
	int colormap_width, colormap_height;
	RGBA* colormap = reinterpret_cast<RGBA*>(stbi_load("colormap_jet.png", &colormap_width, &colormap_height, nullptr, 4));
	CHECK_NOTNULL_F(colormap, "Failed to load colormap: %s", stbi_failure_reason());
	SCOPE_EXIT{ stbi_image_free(colormap); };

	std::vector<RGBA> colors;
	for (float value : data) {
		if (max <= min || !std::isfinite(value)) {
			colors.emplace_back(RGBA{0,0,0,255});
		} else {
			int colormap_x = emath::remap_clamp(value, min, max, 0, colormap_width - 1);
			CHECK_F(0 <= colormap_x && colormap_x < colormap_width);
			colors.emplace_back(colormap[colormap_x]);
		}
	}
	return colors;
}

using namespace emilib;

void show_texture_options(gl::Texture* texture)
{
	auto params = texture->params();
	ImGui::Text("Filter:");
	ImGui::SameLine();
	int is_nearest = params.filter == gl::TexFilter::Nearest;
	ImGui::RadioButton("Nearest", &is_nearest, 1);
	ImGui::SameLine();
	ImGui::RadioButton("Linear", &is_nearest, 0);
	params.filter = is_nearest ? gl::TexFilter::Nearest : gl::TexFilter::Linear;
	texture->set_params(params);
}
