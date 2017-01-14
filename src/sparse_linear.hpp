#pragma once

#include <vector>

using Real = double;

template<typename T>
struct Triplet
{
	int row, col;
	T value;

	Triplet() {}
	Triplet(int row_, int col_, T value_) : row(row_), col(col_), value(value_) {}
};

/// rows == rhs.size(). Construct matrix A from triplets. Solve for x in  A * x = rhs.
std::vector<Real> solve_sparse_linear(
	int                               columns,
	const std::vector<Triplet<Real>>& triplets,
	const std::vector<Real>&          rhs);
