#include "sine_denoise_1d.hpp"

#include <cmath>
#include <random>

#include <emath/math.hpp>
#include <emilib/imgui_helpers.hpp>
#include <emilib/irange.hpp>

#include <field_interpolation/field_interpolation.hpp>
#include <field_interpolation/sparse_linear.hpp>

#include "gui.hpp"

using Vec2List = std::vector<ImVec2>;

void show_1d_denoiser_window()
{
	static int         s_seed           =   0;
	static int         s_resolution     = 512;
	static float       s_noise_y_stddev =   0.1f;
	static size_t      s_num_points     = 256;
	static float       s_amplitude      =   0.5f;
	static float       s_freq           =   10.0f;
	static float       s_chirp_factor   =   2.0f;
	static fi::Weights s_weights = [](){
		fi::Weights weights;
		weights.model_1 = 0;
		weights.model_2 = 10;
		return weights;
	}();

	// ------------------------------------------

	std::default_random_engine rng(s_seed);
	std::normal_distribution<float> y_noise(0.0, s_noise_y_stddev);

	Vec2List points;
	Vec2List gt;

	for (int i : emilib::irange(s_num_points)) {
		float t = emath::remap(i, 0, s_num_points - 1, 0, 1);
		float f = s_freq * (1 + t * s_chirp_factor);
		float y = s_amplitude * std::sin(t * f);
		gt.emplace_back(t, y);
		y += y_noise(rng);
		points.emplace_back(t, y);
	}

	// ------------------------------------------

	fi::LatticeField field{{s_resolution}};

	add_field_constraints(&field, s_weights);

	for (const ImVec2& p : points) {
		float x = emath::remap(p.x, 0, 1, 0, s_resolution - 1.0f);
		add_value_constraint(&field, &x, p.y, s_weights.data_pos);
	}

	const int num_unknowns = s_resolution;
	std::vector<float> solution = solve_sparse_linear_exact(field.eq, num_unknowns);
	if (solution.empty()) { solution.resize(num_unknowns, 0.0f); }

	// ------------------------------------------

	ImGui::SliderInt("resolution",       &s_resolution,     10,  1000);
	ImGui::SliderFloat("noise",          &s_noise_y_stddev,  0,     1);
	ImGuiPP::SliderSize("points",        &s_num_points,      0, 10000);
	ImGui::SliderFloat("s_amplitude",    &s_amplitude,       0,     1);
	ImGui::SliderFloat("s_freq",         &s_freq,            0,     100);
	ImGui::SliderFloat("s_chirp_factor", &s_chirp_factor,    0,    10);
	show_weights(&s_weights);

	ImVec2 canvas_size = ImGui::GetContentRegionAvail();
	ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
	ImGui::InvisibleButton("canvas", canvas_size);
	ImDrawList* draw_list = ImGui::GetWindowDrawList();

	auto canvas_from_field = [=](float t, float y) {
		return ImVec2{
			canvas_pos.x + canvas_size.x * t,
			canvas_pos.y + emath::remap(y, -1, +1, canvas_size.y, 0)
		};
	};

	draw_list->AddLine(canvas_pos, canvas_pos + ImVec2(canvas_size.x, 0), ImColor(1.0f, 1.0f, 1.0f, 0.25f));
	draw_list->AddLine(canvas_pos, canvas_pos + ImVec2(0, canvas_size.y), ImColor(1.0f, 1.0f, 1.0f, 0.25f));
	draw_list->AddLine(canvas_pos + canvas_size, canvas_pos + ImVec2(canvas_size.x, 0), ImColor(1.0f, 1.0f, 1.0f, 0.25f));
	draw_list->AddLine(canvas_pos + canvas_size, canvas_pos + ImVec2(0, canvas_size.y), ImColor(1.0f, 1.0f, 1.0f, 0.25f));

	for (const ImVec2& p : points) {
		draw_list->AddCircleFilled(canvas_from_field(p.x, p.y), 2, ImColor(1.0f, 1.0f, 1.0f, 1.0f));
	}

	Vec2List gt_points;
	for (const ImVec2& point : gt) {
		gt_points.push_back(canvas_from_field(point.x, point.y));
	}

	draw_list->AddPolyline(gt_points.data(), gt_points.size(), ImColor(1.0f, 0.0f, 0.0f, 0.5f), false, 2, true);

	Vec2List solution_points;
	for (auto i : emilib::irange(num_unknowns)) {
		float t = emath::remap(i, 0, num_unknowns - 1, 0, 1);
		solution_points.push_back(canvas_from_field(t, solution[i]));
	}

	draw_list->AddPolyline(solution_points.data(), solution_points.size(), ImColor(1.0f, 1.0f, 1.0f, 1.0f), false, 2, true);
}
