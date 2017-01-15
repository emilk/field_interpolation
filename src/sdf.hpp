#pragma once

#include <vector>

#include "sparse_linear.hpp"

struct Point
{
	float x, y;   // Pos
	float dx, dy; // Normal
};

struct Strengths
{
	float data_pos    = 0.01f; // How much we trust the point positions
	float data_normal = 1.00f; // How much we trust the point normals
	float model_0     = 0.00f; // How much we believe the SDF to be zero (regularization).
	float model_1     = 0.00f; // How much we believe the SDF to be uniform.
	float model_2     = 1.00f; // How much we believe the SDF to be smooth.
};

/// Sparse Ax=b where A is described by `triplets` and `rhs` is b.
struct LinearEquation
{
	std::vector<TripletR> triplets;
	std::vector<Real>     rhs;
};

struct LinearEquationPair
{
	int  column;
	Real value;
};

/// Helper to add a row to the linear equation
inline void add_equation(LinearEquation* eq, float rhs, std::initializer_list<LinearEquationPair> pairs)
{
	int row = eq->rhs.size();
	for (const auto& pair : pairs) {
		eq->triplets.emplace_back(row, pair.column, pair.value);
	}
	eq->rhs.emplace_back(rhs);
}

// Points assumed to be in [0, 1] square.
// returns resolution * resolution distances on success, empty vector on fail.
// The first distances is at the points [0,0], the last at [1, 1].
std::vector<float> generate_sdf(size_t resolution, const std::vector<Point>& points, const Strengths& strengths);
