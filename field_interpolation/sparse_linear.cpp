#include "sparse_linear.hpp"

#include <Eigen/Eigen>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>

#include <loguru.hpp>

namespace field_interpolation {

std::ostream& operator<<(std::ostream& os, const LinearEquation& eq)
{
	const size_t num_rows = eq.rhs.size();
	std::vector<std::vector<Triplet>> row_triplets(num_rows);

	for (const auto& triplet : eq.triplets) {
		row_triplets[triplet.row].push_back(triplet);
	}

	for (size_t row = 0; row < num_rows; ++row) {
		os << eq.rhs[row] << " = ";
		for (size_t triplet_idx = 0; triplet_idx < row_triplets[row].size(); ++triplet_idx) {
			const auto& triplet = row_triplets[row][triplet_idx];
			os << triplet.value << " * x" << triplet.col;
			if (triplet_idx + 1 < row_triplets[row].size()) {
				os  << "  +  ";
			}
		}
		os << "\n";
	}
	return os;
}

void add_equation(
	LinearEquation* eq, Weight weight, Rhs rhs, std::initializer_list<LinearEquationPair> pairs)
{
	if (weight.value == 0) { return; }

	// bool all_zero = rhs == 0;
	bool all_zero = true;
	int row = eq->rhs.size();
	for (const auto& pair : pairs) {
		if (pair.value != 0) {
			eq->triplets.emplace_back(row, pair.column, pair.value * weight.value);
			all_zero = false;
		}
	}
	if (!all_zero) {
		eq->rhs.emplace_back(rhs.value * weight.value);
	}
}

// ----------------------------------------------------------------------------

using VectorXr = Eigen::Matrix<float, Eigen::Dynamic, 1>;
using SparseMatrix = Eigen::SparseMatrix<float>;

SparseMatrix as_sparse_matrix(
	const std::vector<Triplet>& triplets, size_t num_rows, size_t num_columns)
{
	LOG_SCOPE_F(1, "as_sparse_matrix");

#if 0
	std::vector<Eigen::Triplet<float>> eigen_triplets;
	eigen_triplets.reserve(triplets.size());
	for (const auto& triplet : triplets) {
		CHECK_GE_F(triplet.col, 0);
		CHECK_GE_F(triplet.row, 0);
		CHECK_LT_F(triplet.col, num_columns);
		CHECK_LT_F(triplet.row, num_rows);
		if (triplet.value != 0.0f) {
			eigen_triplets.emplace_back(triplet.row, triplet.col, triplet.value);
		}
	}

	SparseMatrix A(num_rows, num_columns);
	A.setFromTriplets(eigen_triplets.begin(), eigen_triplets.end());
#else
	static_assert(sizeof(Triplet) == sizeof(Eigen::Triplet<float>), "");
	SparseMatrix A(num_rows, num_columns);
	const Eigen::Triplet<float>* ptr = reinterpret_cast<const Eigen::Triplet<float>*>(triplets.data());
	A.setFromTriplets(ptr, ptr + triplets.size());
#endif
	A.makeCompressed();
	return A;
}

VectorXr as_eigen_vector(const std::vector<float>& values)
{
	return Eigen::Map<VectorXr>(const_cast<float*>(values.data()), values.size());
}

std::vector<float> as_std_vector(const VectorXr& values)
{
	return std::vector<float>(values.data(), values.data() + values.rows() * values.cols());
}

SparseMatrix make_square(const SparseMatrix& A)
{
	LOG_SCOPE_F(1, "Make AtA");
	SparseMatrix AtA = A.transpose() * A;
	CHECK_EQ_F(AtA.rows(), AtA.cols());
	AtA.makeCompressed();
	return AtA;
}

std::vector<float> solve_sparse_linear(const LinearEquation& eq, int num_columns)
{
	LOG_SCOPE_F(1, "solve_sparse_linear");
	const SparseMatrix A = as_sparse_matrix(eq.triplets, eq.rhs.size(), num_columns);
	const SparseMatrix AtA = make_square(A);
	const VectorXr Atb = A.transpose() * as_eigen_vector(eq.rhs);

	LOG_F(1, "A nnz: %lu (%.3f%%)", A.nonZeros(),
		  100.0f * A.nonZeros() / (A.rows() * A.cols()));
	LOG_F(1, "AtA nnz: %lu (%.3f%%)", AtA.nonZeros(),
		  100.0f * AtA.nonZeros() / (AtA.rows() * AtA.cols()));

	LOG_SCOPE_F(1, "Solve");
	// Resolution 177, 2 x 100k points
	// Eigen::SparseQR<SparseMatrix, Eigen::COLAMDOrdering<int>> solver(AtA); // SLOW
	// Eigen::BiCGSTAB<SparseMatrix> solver(AtA); // 1400 ms
	// Eigen::ConjugateGradient<SparseMatrix, Eigen::Upper> solver(AtA); // 1200 ms
	// Eigen::SparseLU<SparseMatrix, Eigen::COLAMDOrdering<int>> solver(AtA); // 600 ms
	// Eigen::SimplicialLDLT<SparseMatrix> solver(AtA); // 500 ms
	// Eigen::SimplicialCholesky<SparseMatrix> solver(AtA); // 500 ms
	Eigen::SimplicialLLT<SparseMatrix> solver(AtA); // 400 ms

	if (solver.info() != Eigen::Success) {
		LOG_F(WARNING, "solver.compute failed");
		return {};
	}

	VectorXr solution = solver.solve(Atb);

	if (solver.info() != Eigen::Success) {
		// std::string error = solver.lastErrorMessage();
		// LOG_F(WARNING, "solver.solve failed: %s", error.c_str());
		LOG_F(WARNING, "solver.solve failed");
		return {};
	}

	return as_std_vector(solution);
}

std::vector<float> solve_sparse_linear_with_guess(
	const LinearEquation&     eq,
	const std::vector<float>& guess,
	float                     error_tolerance)
{
	LOG_SCOPE_F(1, "solve_sparse_linear_with_guess");

	const SparseMatrix A = as_sparse_matrix(eq.triplets, eq.rhs.size(), guess.size());
	const SparseMatrix AtA = make_square(A);
	const VectorXr Atb = A.transpose() * as_eigen_vector(eq.rhs);

	LOG_SCOPE_F(1, "solveWithGuess");
	Eigen::BiCGSTAB<SparseMatrix> solver(AtA);
	// Eigen::ConjugateGradient<SparseMatrix> solver(AtA);
	solver.setTolerance(error_tolerance);
	const VectorXr solution = solver.solveWithGuess(Atb, as_eigen_vector(guess));

	LOG_F(1, "CG iterations: %lu", solver.iterations());
	LOG_F(1, "CG error:      %f",  solver.error());

	if (solver.info() != Eigen::Success) {
		LOG_F(WARNING, "solver.solveWithGuess failed");
		return {};
	}

	return as_std_vector(solution);
}

/// Break the lattice into tiles, each tile_size^D big.
/// Each tile is solved separately, and the results are combined.
/// The produces a result where the high frequency components are very accurate.
VectorXr tile_solver_square(
	const SparseMatrix&     A_full,
	const VectorXr&         b_full,
	const VectorXr&         guess_full,
	const std::vector<int>& sizes_full,
	int                     tile_size)
{
	LOG_SCOPE_F(1, "tile_solver_square");
	CHECK_GE_F(tile_size, 2);
	CHECK_EQ_F(A_full.rows(), A_full.cols());
	CHECK_EQ_F(A_full.rows(), b_full.rows());
	CHECK_EQ_F(A_full.cols(), guess_full.rows());

	std::vector<int> num_tiles;
	int num_tiles_total = 1;
	int unknowns_per_tile = 1; // tile_size^D
	for (auto size : sizes_full) {
		num_tiles.push_back((size + tile_size - 1) / tile_size);
		num_tiles_total *= num_tiles.back();
		unknowns_per_tile *= tile_size;
	}

	LOG_F(1, "num_tiles_total: %d", num_tiles_total);

	const auto calc_tile_and_index = [=](int full_index) -> std::tuple<int, int> {
		int tile_index = 0;
		int index_in_tile = 0;
		int tile_index_stride = 1;
		int index_in_tile_stride = 1;

		for (int d = 0; d < sizes_full.size(); ++d) {
			int full_x = full_index % sizes_full[d];
			int tile_x = full_x / tile_size;
			int x_in_tile = full_x % tile_size;

			tile_index += tile_x * tile_index_stride;
			index_in_tile += x_in_tile * index_in_tile_stride;

			full_index /= sizes_full[d];
			tile_index_stride *= num_tiles[d];
			index_in_tile_stride *= tile_size;
		}

		CHECK_GE_F(index_in_tile, 0);
		CHECK_LT_F(index_in_tile, unknowns_per_tile);

		return std::make_tuple(tile_index, index_in_tile);
	};

	struct Tile
	{
		std::vector<Eigen::Triplet<float>> triplets;
		VectorXr                           rhs;
	};

	std::vector<Tile> tiles(num_tiles_total);
	for (int tile_index = 0; tile_index < num_tiles_total; ++tile_index) {
		ERROR_CONTEXT("tile_index", tile_index);
		tiles[tile_index].rhs = VectorXr::Zero(unknowns_per_tile);

		// Add small regularization, needed for edge tiles which extends outside of lattice:
		for (int i = 0; i < unknowns_per_tile; ++i) {
			tiles[tile_index].triplets.emplace_back(i, i, 1e-6f);
		}
	}

	for (int full_index = 0; full_index < b_full.size(); ++full_index) {
		ERROR_CONTEXT("full_index", full_index);
		int tile_index, index_in_tile;
		std::tie(tile_index, index_in_tile) = calc_tile_and_index(full_index);
		tiles[tile_index].rhs[index_in_tile] = b_full[full_index];
	}

	for (int k=0; k < A_full.outerSize(); ++k) {
		for (SparseMatrix::InnerIterator it(A_full, k); it; ++it) {
			int row_tile, row_index;
			int col_tile, col_index;

			std::tie(row_tile, row_index) = calc_tile_and_index(it.row());
			std::tie(col_tile, col_index) = calc_tile_and_index(it.col());

			if (row_tile == col_tile) {
				// Equation describing a relationship within the same tile
				tiles[row_tile].triplets.emplace_back(row_index, col_index, it.value());
			} else {
				// Between two tiles. We can't connect them, so we substitute in the value from the initial guesses.
				tiles[row_tile].rhs[row_index] -= it.value() * guess_full[it.col()];
				tiles[col_tile].rhs[col_index] -= it.value() * guess_full[it.row()];
			}
		}
	}

	LOG_SCOPE_F(1, "solving tiles");
	VectorXr solution_full = guess_full;

	int num_failures = 0;

	for (int tile_index = 0; tile_index < num_tiles_total; ++tile_index) {
		const auto& tile = tiles[tile_index];
		SparseMatrix A_tile(unknowns_per_tile, unknowns_per_tile);
		A_tile.setFromTriplets(tile.triplets.begin(), tile.triplets.end());
		A_tile.makeCompressed();

		Eigen::SimplicialLLT<SparseMatrix> solver_tile(A_tile);
		if (solver_tile.info() != Eigen::Success) { num_failures += 1; continue; }
		VectorXr solution_tile = solver_tile.solve(tile.rhs);
		if (solver_tile.info() != Eigen::Success) { num_failures += 1; continue; }

		CHECK_GE_F(solution_tile.size(), unknowns_per_tile);

		for (int index_in_tile = 0; index_in_tile < solution_tile.size(); ++index_in_tile) {
			int tile_index_copy = tile_index;
			int index_in_tile_copy = index_in_tile;
			int stride = 1;
			int full_index = 0;
			bool inside = true;

			for (int d = 0; d < sizes_full.size(); ++d) {
				int tile_x = tile_index_copy % num_tiles[d];
				int x_in_tile = index_in_tile_copy % tile_size;
				int full_x = tile_x * tile_size + x_in_tile;

				inside &= (full_x < sizes_full[d]);

				full_index += full_x * stride;
				tile_index_copy /= num_tiles[d];
				index_in_tile_copy /= tile_size;
				stride *= sizes_full[d];
			}

			if (inside) {
				solution_full[full_index] = solution_tile[index_in_tile];
			}
		}
	}

	LOG_IF_F(WARNING, num_failures > 0, "%d/%d tiles failed", num_failures, num_tiles_total);

	return solution_full;
}

std::vector<float> solve_tiled_with_guess(
	const LinearEquation&     eq,
	const std::vector<float>& guess_vec,
	const std::vector<int>&   sizes,
	const SolveOptions&       options)
{
	LOG_SCOPE_F(1, "solve_tiled_with_guess");
	int num_unknowns = 1;
	for (auto size : sizes) { num_unknowns *= size; }

	if (guess_vec.size() != num_unknowns)
	{
		LOG_F(ERROR, "Incomplete guess.");
		return {};
	}

	const VectorXr b = as_eigen_vector(eq.rhs);

	const SparseMatrix A = as_sparse_matrix(eq.triplets, eq.rhs.size(), num_unknowns);
	const SparseMatrix AtA = make_square(A);
	const VectorXr Atb = A.transpose() * b;

	LOG_F(1, "A nnz:   %lu (%.3f%%)", A.nonZeros(),
		  100.0f * A.nonZeros() / (A.rows() * A.cols()));

	LOG_F(1, "AtA nnz: %lu (%.3f%%)", AtA.nonZeros(),
		  100.0f * AtA.nonZeros() / (AtA.rows() * AtA.cols()));

	// ------------------------------------------------------------------------

	VectorXr guess = as_eigen_vector(guess_vec);

	if (options.tile) {
		guess = tile_solver_square(AtA, Atb, guess, sizes, options.tile_size);
	}

	if (!options.cg) {
		return as_std_vector(guess);
	}

	LOG_SCOPE_F(1, "solveWithGuess");
	Eigen::BiCGSTAB<SparseMatrix> solver(AtA);
	// Eigen::ConjugateGradient<SparseMatrix> solver(AtA);

	solver.setTolerance(options.error_tolerance);
	const VectorXr solution = solver.solveWithGuess(Atb, guess);

	LOG_F(1, "CG iterations: %lu", solver.iterations());
	LOG_F(1, "CG error:      %f",  solver.error());

	if (solver.info() != Eigen::Success) {
		LOG_F(WARNING, "solver.solveWithGuess failed");
		return as_std_vector(guess);
	}

	return as_std_vector(solution);
}

} // namespace field_interpolation
