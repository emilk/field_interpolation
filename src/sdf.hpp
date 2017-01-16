#pragma once

#include <vector>

#include "sparse_linear.hpp"

/*
This library helps you set up a sparse linear system of to which the solution
is an implicit function whose iso-surface at threshold 0 approximates the given constraints.

In particular, if you have a set of points on the surface of an object,
you can use this library to generate a surface like this:

	LinearEquation eq;
	add_model_constraints(&eq, width, height, strengths);
	for (const auto& point : my_points) {
		add_point_constraint(&eq, width, height, strengths, point.pos, point.normal);
	}

	const int num_unknowns = width * height;
	const auto sdf = solve_sparse_linear(num_unknowns, eq.triplets, eq.rhs);
	CHECK_EQ_F(sdf.size(), num_unknowns);

	bool double_precision = false;
	return marching_squares(width, height, sdf.data(), double_precision);

The lattice coordinates go from [0, 0] to [width - 1, height - 1] (inclusive).
*/

struct Strengths
{
	float data_pos    = 1.00f; // How much we trust the point positions
	float data_normal = 1.00f; // How much we trust the point normals
	float model_0     = 0.00f; // How much we believe the SDF to be zero (regularization).
	float model_1     = 0.00f; // How much we believe the SDF to be uniform.
	float model_2     = 1.00f; // How much we believe the SDF to be smooth.
	float model_3     = 0.00f; // Another order of smoothness.
};

/// Sparse Ax=b where A is described by `triplets` and `rhs` is b.
struct LinearEquation
{
	std::vector<Triplet> triplets;
	std::vector<float>   rhs;
};

struct LinearEquationPair
{
	int   column;
	float value;
};

/// Helper to add a row to the linear equation.
inline void add_equation(LinearEquation* eq, float rhs, std::initializer_list<LinearEquationPair> pairs)
{
	int row = eq->rhs.size();
	for (const auto& pair : pairs) {
		eq->triplets.emplace_back(row, pair.column, pair.value);
	}
	eq->rhs.emplace_back(rhs);
}

/// Like add_equation, but will be a no-op if all arguments are zero, or any index is outside [0, num_unknowns) range
inline void add_equation_checked(LinearEquation* eq, int num_unknowns, float rhs, std::initializer_list<LinearEquationPair> pairs)
{
	bool all_zero = rhs == 0.0f;
	for (const auto& pair : pairs) {
		if (pair.column < 0 || num_unknowns <= pair.column) { return; }
		all_zero &= pair.value == 0.0f;
	}
	if (all_zero) { return; }
	add_equation(eq, rhs, pairs);
}

/// Add equations describing the model: a smooth field on a lattice.
void add_model_constraints(LinearEquation* eq, int width, int height, const Strengths& strengths);

/// Add equations describing the vicinity of a particle.
void add_point_constraint(
	LinearEquation*  eq,
	int              width,
	int              height,
	const Strengths& strengths,
	const float      pos[2],
	const float      normal[2]);
