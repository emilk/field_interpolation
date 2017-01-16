#include "sdf.hpp"

#include <cmath>

#include <loguru.hpp>

void add_point_position_constraint(
	LinearEquation* eq,
	int             width,
	int             height,
	float           strength,
	const float     pos[2])
{
	int num_unknowns = width * height;
	auto index = [=](int x, int y) -> int { return y * width + x; };

	const int x_floored = std::floor(pos[0]);
	const int y_floored = std::floor(pos[1]);

	// Bilinear interpolation of points influence:
	const float tx = pos[0] - x_floored;
	const float ty = pos[1] - y_floored;

	add_equation_checked(eq, num_unknowns, 0.0f, {
		{index(x_floored, y_floored), strength * (1 - tx) * (1 - ty)},
	});

	add_equation_checked(eq, num_unknowns, 0.0f, {
		{index(x_floored + 1, y_floored), strength * tx * (1 - ty)},
	});

	add_equation_checked(eq, num_unknowns, 0.0f, {
		{index(x_floored, y_floored + 1), strength * (1 - tx) * ty},
	});

	add_equation_checked(eq, num_unknowns, 0.0f, {
		{index(x_floored + 1, y_floored + 1), strength * tx * ty},
	});
}

void add_point_normal_constraint(
	LinearEquation* eq,
	int             width,
	int             height,
	float           strength,
	const float     pos[2],
	const float     normal[2])
{
	int num_unknowns = width * height;
	auto index = [=](int x, int y) -> int { return y * width + x; };

	const int x_floored = std::floor(pos[0]);
	const int y_floored = std::floor(pos[1]);

	// Influence delta between the two closest cells on each axis:
	const int x_rounded = std::round(pos[0]);
	const int y_rounded = std::round(pos[1]);

	// d f(x, y) / dx = point.dx
	add_equation_checked(eq, num_unknowns, normal[0] * strength, {
		{index(x_floored + 0, y_rounded), -strength},
		{index(x_floored + 1, y_rounded), +strength},
	});

	// d f(x, y) / dy = point.dy
	add_equation_checked(eq, num_unknowns, normal[1] * strength, {
		{index(x_rounded, y_floored + 0), -strength},
		{index(x_rounded, y_floored + 1), +strength},
	});
}

void add_point_constraint(
	LinearEquation*  eq,
	int              width,
	int              height,
	const Strengths& strengths,
	const float      pos[2],
	const float      normal[2])
{
	add_point_position_constraint(eq, width, height, strengths.data_pos, pos);
	add_point_normal_constraint(eq, width, height, strengths.data_normal, pos, normal);
}

/// Add smoothness constraints between the unknowns:  index - stride, index, index + stride, ...
void add_model_constraint(
	LinearEquation*  eq,
	int              num_unknowns,
	const Strengths& strengths,
	int              index,
	int              stride)
{
	// f(x) = 0
	add_equation_checked(eq, num_unknowns, 0.0f, {
		{index, strengths.model_0},
	});

	// f'(x) = 0   ⇔   f(x) = f(x + 1)
	add_equation_checked(eq, num_unknowns, 0.0f, {
		{index + 0,      +strengths.model_1},
		{index + stride, +strengths.model_1},
	});

	// f''(x) = 0   ⇔   f'(x - ½) = f'(x + ½)
	add_equation_checked(eq, num_unknowns, 0.0f, {
		{index - stride, +1 * strengths.model_2},
		{index + 0,      -2 * strengths.model_2},
		{index + stride, +1 * strengths.model_2},
	});

	// f'''(x) = 0   ⇔   f''(x - 1) = f''(x + 1)
	add_equation_checked(eq, num_unknowns, 0.0f, {
		{index - 2 * stride, +1 * strengths.model_3},
		{index - 1 * stride, -2 * strengths.model_3},
		{index + 1 * stride, +2 * strengths.model_3},
		{index + 2 * stride, -1 * strengths.model_3},
	});
}

void add_model_constraints(LinearEquation* eq, int width, int height, const Strengths& strengths)
{
	const auto num_unknowns = width * height;
	for (int index = 0; index < num_unknowns; ++index) {
		add_model_constraint(eq, width * height, strengths, index, 1);
		add_model_constraint(eq, width * height, strengths, index, width);
	}
}
