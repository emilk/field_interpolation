#pragma once

#include <vector>

struct Triplet
{
	int row, col;
	float    value;

	Triplet() {}
	Triplet(int row_, int col_, float value_) : row(row_), col(col_), value(value_) {}
};

/// Solve a sparse linear least squared problem.
/// Construct matrix A from triplets. Solve for x in  A * x = rhs.
/// `rows` == `rhs.size()`.
/// `columns` == number of unknowns
std::vector<float> solve_sparse_linear(
	int                         columns,
	const std::vector<Triplet>& triplets,
	const std::vector<float>&   rhs,
	bool                        double_precision);
