#include "field_1d.hpp"

#include <sstream>

#include <visit_struct/visit_struct.hpp>

#include <configuru.hpp>
#include <emilib/file_system.hpp>
#include <emilib/imgui_helpers.hpp>
#include <emilib/irange.hpp>

#include "gui.hpp"

struct Point1D
{
	float pos, value, gradient;
};
VISITABLE_STRUCT(Point1D, pos, value, gradient);

struct Field1DInput
{
	std::vector<Point1D> points{
		{0.2f, 0, +1},
		{0.8f, 0, -1},
	};

	int resolution = 12;
	fi::Weights weights;
};
VISITABLE_STRUCT(Field1DInput, points, resolution, weights);

void log_configuru_errors(const std::string& msg)
{
	LOG_F(ERROR, "%s", msg.c_str());
}

Field1DInput load_1d_field()
{
	Field1DInput input;
	if (fs::file_exists("1d_field.json")) {
		const auto config = configuru::parse_file("1d_field.json", configuru::JSON);
		configuru::deserialize(&input, config, log_configuru_errors);
	}
	return input;
}

bool show_options(Field1DInput* input)
{
	bool changed = false;

	changed |= ImGui::SliderInt("resolution", &input->resolution, 4, 512);
	changed |= show_weights(&input->weights);

	for (const auto i : emilib::indices(input->points)) {
		auto& point = input->points[i];
		ImGui::PushID(i);
		ImGui::PushItemWidth(ImGui::GetWindowContentRegionWidth() * 0.25f);
		ImGui::Text("Point %lu:", i);
		ImGui::SameLine();
		changed |= ImGui::SliderFloat("pos", &point.pos, 0, 1);
		ImGui::SameLine();
		changed |= ImGui::SliderFloat("value", &point.value, 0, 1);
		ImGui::SameLine();
		changed |= ImGui::SliderFloat("gradient", &point.gradient, -1, 1);
		ImGui::PopItemWidth();
		ImGui::PopID();
	}

	if (input->points.size() >= 2 && ImGui::Button("Remove point")) {
		input->points.pop_back();
		changed = true;
		ImGui::SameLine();
	}
	if (ImGui::Button("Add point")) {
		auto point = input->points.back();
		input->points.push_back(point);
		changed = true;
	}

	return changed;
}

void show_field_equations(const fi::LatticeField& field)
{
	std::stringstream ss;
	ss << field.eq;
	std::string eq_str = ss.str();
	ImGui::Text("%lu equations:\n", field.eq.rhs.size());
	ImGui::TextUnformatted(eq_str.c_str());
}

void show_1d_field_window_for(Field1DInput* input)
{
	if (show_options(input)) {
		configuru::dump_file("1d_field.json", configuru::serialize(*input), configuru::JSON);
	}

	fi::LatticeField field{{input->resolution}};

	for (const auto& point : input->points) {
		float pos_lattice = point.pos * (input->resolution - 1);
		float gradient_lattice = point.gradient / (input->resolution - 1);
		add_value_constraint(&field, &pos_lattice, point.value, input->weights.data_pos);
		add_gradient_constraint(&field, &pos_lattice, &gradient_lattice, input->weights.data_gradient, input->weights.gradient_kernel);
	}

	add_field_constraints(&field, input->weights);

	const size_t num_unknowns = input->resolution;
	auto interpolated = solve_sparse_linear(field.eq, num_unknowns);
	if (interpolated.size() != num_unknowns) {
		LOG_F(ERROR, "Failed to find a solution");
		interpolated.resize(num_unknowns, 0.0f);
	}
	ImGui::Text("interpolated: %f %f ...", interpolated[0], interpolated[1]);

	ImVec2 canvas_size{384, 384};
	ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
	ImGui::InvisibleButton("canvas", canvas_size);
	ImDrawList* draw_list = ImGui::GetWindowDrawList();

	auto canvas_from_field = [=](float x, float y) {
		return ImVec2{
			canvas_pos.x + canvas_size.x * x,
			canvas_pos.y + canvas_size.y * (1 - y)};
	};

	for (size_t i : emilib::irange_inclusive<size_t>(0, 1)) {
		draw_list->AddLine(canvas_from_field(0, i), canvas_from_field(1, i), ImColor(1.0f, 1.0f, 1.0f, 0.25f));
		draw_list->AddLine(canvas_from_field(i, 0), canvas_from_field(i, 1), ImColor(1.0f, 1.0f, 1.0f, 0.25f));
	}

	std::vector<ImVec2> field_points;
	for (size_t i : emilib::indices(interpolated)) {
		field_points.push_back(canvas_from_field(i / (input->resolution - 1.0f), interpolated[i]));
		draw_list->AddCircleFilled(field_points.back(), 2, ImColor(1.0f, 1.0f, 1.0f, 1.0f));
	}

	draw_list->AddPolyline(field_points.data(), field_points.size(), ImColor(1.0f, 1.0f, 1.0f, 1.0f), false, 2, true);

	for (const auto& point : input->points) {
		float arrow_len = 16;
		ImVec2 point_pos = canvas_from_field(point.pos, point.value);
		draw_list->AddCircleFilled(point_pos, 5, ImColor(1.0f, 0.0f, 0.0f, 1.0f));
		if (input->weights.data_gradient > 0) {
			ImVec2 gradient_offset = {arrow_len, -point.gradient * arrow_len};
			draw_list->AddLine(point_pos - gradient_offset, point_pos + gradient_offset, ImColor(1.0f, 0.0f, 0.0f, 0.5f), 2);
		}
	}

	show_field_equations(field);
}

void show_1d_field_window()
{
	static auto s_input = load_1d_field();
	show_1d_field_window_for(&s_input);
}
