#pragma once

#include <iosfwd>
#include <vector>

namespace field_interpolation {

struct Triplet
{
	int row, col;
	float value;

	Triplet() {}
	Triplet(int row_, int col_, float value_) : row(row_), col(col_), value(value_) {}
};

/// Sparse Ax=b and AᵀAx=Aᵀb
struct LinearEquation
{
	// Used for calculating the error per equation/row:
	std::vector<Triplet> A;
	std::vector<float>   b;

	// Used for solving the least squares problem:
	std::vector<Triplet> AtA;
	std::vector<double>  Atb;

	int num_cols() const { return Atb.size(); }
	int num_rows() const { return b.size(); }

	LinearEquation() = default;
	LinearEquation(int num_unknowns) : Atb(num_unknowns, 0.0f) {}
};

std::ostream& operator<<(std::ostream& os, const LinearEquation& eq);

struct LinearEquationPair
{
	int   column;
	float value;
};

struct Weight { float value; };
struct Rhs    { float value; };

/// Helper to add a row to the linear equation.
void add_equation_impl(
	LinearEquation* eq, Weight weight, Rhs rhs, const LinearEquationPair* pairs, int num_pairs);

/// Helper to add a row to the linear equation.
inline void add_equation(
	LinearEquation* eq, Weight weight, Rhs rhs, std::initializer_list<LinearEquationPair> pairs)
{
	add_equation_impl(eq, weight, rhs, pairs.begin(), pairs.size());
}

/// Solve a sparse linear least squared problem.
/// Construct matrix A from triplets. Solve for x in  A * x = rhs.
/// `rows` == `rhs.size()`.
/// `num_columns` == number of unknowns
/// Duplicate elements in triplets will be summed.
std::vector<float> solve_sparse_linear(const LinearEquation& eq, int num_columns);

/// Least square solving for x in Ax = rhs using an iterative Conjugate Gradient solver.
/// `guess` is a starting guess for x.
std::vector<float> solve_sparse_linear_with_guess(
	const LinearEquation&     eq,
	const std::vector<float>& guess,
	int                       max_iterations,   ///< Set to zero for default (problem size).
	float                     error_tolerance); ///< Set to zero for default (float epsilon).

struct SolveOptions
{
	bool  tile             = true;    ///< Break up problem into tiles and solve each tile exactly?
	int   tile_size        = 16;      ///< Side of each tile (tile_size^D unknowns).
	bool  cg               = true;    ///< Follow tile phase with a CG phase?
	int   max_iterations   = 0;       ///< Set to zero for default (problem size).
	float error_tolerance  =  1e-3f;  ///< Set to zero for default (float epsilon).
};

/// Approximate solver using a given guess + tiled solver + conjugate gradient.
std::vector<float> solve_tiled_with_guess(
	const LinearEquation&     eq,
	const std::vector<float>& guess,
	const std::vector<int>&   sizes,
	const SolveOptions&       options);

} // namespace field_interpolation
