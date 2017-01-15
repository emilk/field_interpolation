#include "sdf.hpp"

#include <cmath>

#include <loguru.hpp>

void add_point_constraint(
	LinearEquation*  eq,
	size_t           resolution,
	const Strengths& strengths,
	const Point&     point)
{
	CHECK_NOTNULL_F(eq);
	ERROR_CONTEXT("row", eq->rhs.size());

	auto index = [=](int x, int y) -> int { return y * resolution + x; };

	const int x_floored = std::floor(point.x);
	const int y_floored = std::floor(point.y);

	if (x_floored < 0 || resolution <= x_floored) { return; }
	if (y_floored < 0 || resolution <= y_floored) { return; }

	if (0.0f < strengths.data_pos) {
		// Bilinear interpolation of points influence:
		const float tx = point.x - x_floored;
		const float ty = point.y - y_floored;

		add_equation(eq, 0.0f, {{index(x_floored, y_floored), strengths.data_pos * (1 - tx) * (1 - ty)}});

		if (x_floored + 1 < resolution) {
			add_equation(eq, 0.0f, {{index(x_floored + 1, y_floored), strengths.data_pos * tx * (1 - ty)}});
		}

		if (y_floored + 1 < resolution) {
			add_equation(eq, 0.0f, {{index(x_floored, y_floored + 1), strengths.data_pos * (1 - tx) * ty}});
		}

		if (x_floored + 1 < resolution && y_floored + 1 < resolution) {
			add_equation(eq, 0.0f, {{index(x_floored + 1, y_floored + 1), strengths.data_pos * tx * ty}});
		}
	}

	if (0.0f < strengths.data_normal) {
		// Influence delta between the two closest cells on each axis:
		const int x_rounded = std::round(point.x);
		const int y_rounded = std::round(point.y);

		if (0 <= x_floored && x_floored + 1 < resolution
		 && 0 <= y_rounded && y_rounded < resolution) {
			// d f(x, y) / dx = point.dx
			add_equation(eq, point.dx * strengths.data_normal, {
				{index(x_floored + 0, y_rounded), -strengths.data_normal},
				{index(x_floored + 1, y_rounded), +strengths.data_normal}});
		}

		if (0 <= x_rounded && x_rounded < resolution
		 && 0 <= y_floored && y_floored + 1 < resolution) {
			// d f(x, y) / dy = point.dy
			add_equation(eq, point.dy * strengths.data_normal, {
				{index(x_rounded, y_floored + 0), -strengths.data_normal},
				{index(x_rounded, y_floored + 1), +strengths.data_normal}});
		}
	}
}

void add_model_constraint(
	LinearEquation*  eq,
	size_t           resolution,
	const Strengths& strengths,
	int              x,
	int              y)
{
	auto index = [=](int x, int y) -> int { return y * resolution + x; };

	if (strengths.model_0 > 0.0f) {
		add_equation(eq, 0.0f, {{index(x, y), strengths.model_0}});
	}

	if (strengths.model_1 > 0.0f) {
		if (0 < x) {
			// df/dx = 0
			add_equation(eq, 0.0f, {
				{index(x - 1, y), -strengths.model_1},
				{index(x + 0, y), +strengths.model_1},
			});
		}

		if (0 < y) {
			// df/dy = 0
			add_equation(eq, 0.0f, {
				{index(x, y - 1), -strengths.model_1},
				{index(x, y + 0), +strengths.model_1},
			});
		}
	}

	if (strengths.model_2 > 0.0f) {
		if (0 < x && x + 1 < resolution) {
			// d2f/dx2 = 0
			add_equation(eq, 0.0f, {
				{index(x - 1, y), +1 * strengths.model_2},
				{index(x + 0, y), -2 * strengths.model_2},
				{index(x + 1, y), +1 * strengths.model_2},
			});
		}

		if (0 < y && y + 1 < resolution) {
			// d2f/dy2 = 0
			add_equation(eq, 0.0f, {
				{index(x, y - 1), +1 * strengths.model_2},
				{index(x, y + 0), -2 * strengths.model_2},
				{index(x, y + 1), +1 * strengths.model_2},
			});
		}
	}
}

std::vector<float> generate_sdf(
	size_t resolution, const std::vector<Point>& points, const Strengths& strengths, bool double_precision)
{
	ERROR_CONTEXT("resolution", resolution);
	ERROR_CONTEXT("points", points.size());
	LOG_SCOPE_F(INFO, "generate_sdf");
	LinearEquation eq;

	// Data constraints:
	for (const auto& point : points) {
		add_point_constraint(&eq, resolution, strengths, point);
	}

	// Model constraints:
	for (int y = 0; y < resolution; ++y) {
		for (int x = 0; x < resolution; ++x) {
			add_model_constraint(&eq, resolution, strengths, x, y);
		}
	}

	LOG_F(INFO, "%lu equations", eq.rhs.size());
	LOG_F(INFO, "%lu values in matrix", eq.triplets.size());

	LOG_SCOPE_F(INFO, "solve_sparse_linear");
	return solve_sparse_linear(resolution * resolution, eq.triplets, eq.rhs, double_precision);
}
