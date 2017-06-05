#include "line_2d.hpp"

#include <cmath>
#include <random>

#include <emath/math.hpp>
#include <emilib/imgui_helpers.hpp>
#include <emilib/irange.hpp>

#include <field_interpolation/sparse_linear.hpp>

#include "gui.hpp"

void show_line_2d()
{
	static size_t s_seed           =   0;
	static int    s_n              = 512;
	static float  s_noise          =   0.05f;
	static float  s_regularization =   1.0f;
	static float  s_smoothness_0   =   0.0f;
	static float  s_smoothness_1   = 100.0f;
	static float  s_smoothness_2   =   0.0f;
	static float  s_smoothness_3   =   0.0f;

	ImGui::SliderInt("N",                &s_n,              10,       1000);
	ImGui::SliderFloat("Noise",          &s_noise,           0,    2, "%.3f", 4);
	ImGui::SliderFloat("Regularization", &s_regularization,  0, 1000, "%.3f", 4);
	ImGui::SliderFloat("f'(0) = 0 (C0 smoothness)",   &s_smoothness_0, 0, 1000, "%.3f", 4);
	ImGui::SliderFloat("f''(0) = 0 (C1 smoothness)",   &s_smoothness_1, 0, 1000, "%.3f", 4);
	ImGui::SliderFloat("f'''(0) = 0 (C2 smoothness)",  &s_smoothness_2, 0, 1000, "%.3f", 4);
	ImGui::SliderFloat("f''''(0) = 0 (C3 smoothness)", &s_smoothness_3, 0, 1000, "%.3f", 4);

	// ------------------------------------------------------------------------

	std::default_random_engine rng(s_seed);
	std::normal_distribution<float> random_normal;

	Vec2List in_points;
	for (const int i : emilib::irange(s_n)) {
		float a = emath::remap(i, 0, s_n - 1, 0.0f, 2 * emath::TAUf);
		float r = emath::remap(i, 0, s_n - 1, 0.1f, 0.9f);
		float x = r * std::cos(a) + s_noise * random_normal(rng);
		float y = r * std::sin(a) + s_noise * random_normal(rng);
		in_points.push_back({x, y});
	}

	// ------------------------------------------------------------------------

	fi::LinearEquation eq;

	for (const int i : emilib::indices(in_points)) {
		fi::add_equation(&eq, fi::Weight{s_regularization}, fi::Rhs{in_points[i].x}, {
			{2 * i + 0, 1.0f}
		});
		fi::add_equation(&eq, fi::Weight{s_regularization}, fi::Rhs{in_points[i].y}, {
			{2 * i + 1, 1.0f}
		});
	}

	for (const int i : emilib::irange(in_points.size() - 1)) {
		for (const int d : emilib::irange(2)) {
			fi::add_equation(&eq, fi::Weight{s_smoothness_0}, fi::Rhs{0.0f}, {
				{2 * (i + 0) + d, +1.0f},
				{2 * (i + 1) + d, -1.0f},
			});
		}
	}

	for (const int i : emilib::irange(in_points.size() - 2)) {
		for (const int d : emilib::irange(2)) {
			fi::add_equation(&eq, fi::Weight{s_smoothness_1}, fi::Rhs{0.0f}, {
				{2 * (i + 0) + d, +1.0f},
				{2 * (i + 1) + d, -2.0f},
				{2 * (i + 2) + d, +1.0f},
			});
		}
	}

	for (const int i : emilib::irange(in_points.size() - 3)) {
		for (const int d : emilib::irange(2)) {
			fi::add_equation(&eq, fi::Weight{s_smoothness_2}, fi::Rhs{0.0f}, {
				{2 * (i + 0) + d, +1.0f},
				{2 * (i + 1) + d, -3.0f},
				{2 * (i + 2) + d, +3.0f},
				{2 * (i + 3) + d, -1.0f},
			});
		}
	}

	for (const int i : emilib::irange(in_points.size() - 4)) {
		for (const int d : emilib::irange(2)) {
			fi::add_equation(&eq, fi::Weight{s_smoothness_3}, fi::Rhs{0.0f}, {
				{2 * (i + 0) + d, +1.0f},
				{2 * (i + 1) + d, -4.0f},
				{2 * (i + 2) + d, +6.0f},
				{2 * (i + 3) + d, -4.0f},
				{2 * (i + 4) + d, +1.0f},
			});
		}
	}

	// ------------------------------------------------------------------------

	const auto solution = solve_sparse_linear_exact(eq, 2 * in_points.size());
	Vec2List out_points;
	for (const int i : emilib::irange(solution.size() / 2)) {
		out_points.push_back({solution[2 * i + 0], solution[2 * i + 1]});
	}

	// ------------------------------------------------------------------------

	const ImColor kBeforeColor = ImColor(1.0f, 0.0f, 0.0f, 1.0f);
	const ImColor kAfterColor  = ImColor(1.0f, 1.0f, 1.0f, 1.0f);

	ImGui::Text("Task: smooth the input line while minimizing the L2 point-to-point distances.");
	ImGui::TextColored(kBeforeColor, "Noisy input");
	ImGui::TextColored(kAfterColor, "Smoothed output");

	ImVec2 canvas_size = ImGui::GetContentRegionAvail();
	canvas_size.x = canvas_size.y = std::min(canvas_size.x, canvas_size.y);
	ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
	ImGui::InvisibleButton("canvas", canvas_size);
	ImDrawList* draw_list = ImGui::GetWindowDrawList();

	auto canvas_from_field = [=](ImVec2 p) {
		return ImVec2{
			canvas_pos.x + emath::remap(p.x, -1, +1, 0, canvas_size.x),
			canvas_pos.y + emath::remap(p.y, -1, +1, 0, canvas_size.y),
		};
	};

	for (size_t i : emilib::irange(in_points.size() - 1)) {
		auto p0 = in_points[i];
		auto p1 = in_points[i + 1];
		draw_list->AddLine(canvas_from_field(p0), canvas_from_field(p1), kBeforeColor);
	}

	for (size_t i : emilib::irange(out_points.size() - 1)) {
		auto p0 = out_points[i];
		auto p1 = out_points[i + 1];
		draw_list->AddLine(canvas_from_field(p0), canvas_from_field(p1), kAfterColor);
	}
}
