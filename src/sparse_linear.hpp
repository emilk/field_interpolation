#pragma once

#include <vector>

struct Triplet
{
	int row, col;
	float value;

	Triplet() {}
	Triplet(int row_, int col_, float value_) : row(row_), col(col_), value(value_) {}
};

/// Solve a sparse linear least squared problem.
/// Construct matrix A from triplets. Solve for x in  A * x = rhs.
/// `rows` == `rhs.size()`.
/// `num_columns` == number of unknowns
/// Duplicate elements in triplets will be summed.
std::vector<float> solve_sparse_linear(
	int                         num_columns,
	const std::vector<Triplet>& triplets,
	const std::vector<float>&   rhs);

/// Least square solving for x in Ax = rhs.
/// `guess` is a starting guess for x.
std::vector<float> solve_sparse_linear_with_guess(
	const std::vector<Triplet>& triplets,
	const std::vector<float>&   rhs,
	const std::vector<float>&   guess);
