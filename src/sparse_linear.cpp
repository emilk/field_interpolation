#include "sparse_linear.hpp"

#include <Eigen/Eigen>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>

#include <loguru.hpp>

using VectorXr = Eigen::Matrix<float, Eigen::Dynamic, 1>;
using SparseMatrix = Eigen::SparseMatrix<float>;

SparseMatrix as_sparse_matrix(
    const std::vector<Triplet>& triplets, size_t num_rows, size_t num_columns)
{
	LOG_SCOPE_F(INFO, "as_sparse_matrix");

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
	LOG_SCOPE_F(INFO, "AtA");
	SparseMatrix AtA = A.transpose() * A;
	CHECK_EQ_F(AtA.rows(), AtA.cols());
	AtA.makeCompressed();
	return AtA;
}

std::vector<float> solve_sparse_linear(
	int                         num_columns,
	const std::vector<Triplet>& triplets,
	const std::vector<float>&   rhs)
{
	LOG_SCOPE_F(INFO, "solve_sparse_linear");
	const SparseMatrix A = as_sparse_matrix(triplets, rhs.size(), num_columns);
	const SparseMatrix AtA = make_square(A);
	const VectorXr Atb = A.transpose() * as_eigen_vector(rhs);

	LOG_F(INFO, "A nnz: %lu (%.3f%%)", A.nonZeros(),
	      100.0f * A.nonZeros() / (A.rows() * A.cols()));
	LOG_F(INFO, "AtA nnz: %lu (%.3f%%)", AtA.nonZeros(),
	      100.0f * AtA.nonZeros() / (AtA.rows() * AtA.cols()));

	LOG_SCOPE_F(INFO, "Solve");
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
	// VectorXr solution = solver.solve(as_eigen_vector(rhs));

	if (solver.info() != Eigen::Success) {
		// std::string error = solver.lastErrorMessage();
		// LOG_F(WARNING, "solver.solve failed: %s", error.c_str());
		LOG_F(WARNING, "solver.solve failed");
		return {};
	}

	return as_std_vector(solution);
}

std::vector<float> solve_sparse_linear_with_guess(
	const std::vector<Triplet>& triplets,
	const std::vector<float>&   rhs,
	const std::vector<float>&   guess,
	float                       error_tolerance)
{
	LOG_SCOPE_F(INFO, "solve_sparse_linear_with_guess");

	const SparseMatrix A = as_sparse_matrix(triplets, rhs.size(), guess.size());
	const SparseMatrix AtA = make_square(A);
	const VectorXr Atb = A.transpose() * as_eigen_vector(rhs);

	LOG_SCOPE_F(INFO, "solveWithGuess");
	Eigen::BiCGSTAB<SparseMatrix> solver(AtA);
	solver.setTolerance(error_tolerance);
	const VectorXr solution = solver.solveWithGuess(Atb, as_eigen_vector(guess));

	LOG_F(INFO, "CG iterations: %lu", solver.iterations());
	LOG_F(INFO, "CG error:      %f",  solver.error());

	if (solver.info() != Eigen::Success) {
		LOG_F(WARNING, "solver.solveWithGuess failed");
		return {};
	}

	return as_std_vector(solution);
}

VectorXr downscale_solver(
	const SparseMatrix&     AtA_full,
	const VectorXr&         Atb_full,
	const std::vector<int>& sizes_full,
	int                     downscale_factor)
{
	LOG_SCOPE_F(INFO, "downscale_solver");
	CHECK_GE_F(downscale_factor, 2);

	std::vector<int> sizes_small;
	int num_unknowns_small = 1;
	for (int size_full : sizes_full) {
		sizes_small.push_back((size_full + downscale_factor - 1) / downscale_factor);
		num_unknowns_small *= sizes_small.back();
	}

	const auto as_small_index = [=](int full_index) {
		int index_small = 0;
		int stride_small = 1;
		for (int d = 0; d < sizes_full.size(); ++d) {
			int pos_full = full_index % sizes_full[d];
			int pos_small = pos_full / downscale_factor;
			index_small += stride_small * pos_small;
			full_index /= sizes_full[d];
			stride_small *= sizes_small[d];
		}
		return index_small;
	};

	std::vector<Eigen::Triplet<float>> AtA_triplets_small;
	for (int k=0; k < AtA_full.outerSize(); ++k) {
		for (SparseMatrix::InnerIterator it(AtA_full, k); it; ++it) {
			AtA_triplets_small.emplace_back(as_small_index(it.row()), as_small_index(it.col()), it.value());
		}
	}

	SparseMatrix AtA_small(num_unknowns_small, num_unknowns_small);
	AtA_small.setFromTriplets(AtA_triplets_small.begin(), AtA_triplets_small.end());
	AtA_small.makeCompressed();

	VectorXr Atb_small = VectorXr::Zero(num_unknowns_small);
	for (int full_i = 0; full_i < Atb_full.size(); ++full_i) {
		Atb_small[as_small_index(full_i)] += Atb_full[full_i];
	}

	// ------------------------------------------------------------------------

	LOG_F(INFO, "AtA_small nnz: %lu (%.3f%%)", AtA_small.nonZeros(),
	      100.0f * AtA_small.nonZeros() / (AtA_small.rows() * AtA_small.cols()));

	Eigen::SimplicialLLT<SparseMatrix> solver_small(AtA_small);

	if (solver_small.info() != Eigen::Success) {
		LOG_F(WARNING, "solver_small.compute failed");
		return {};
	}

	VectorXr solution_small = solver_small.solve(Atb_small);

	if (solver_small.info() != Eigen::Success) {
		LOG_F(WARNING, "solver_small.solve failed");
		return {};
	}

	VectorXr guess_full = VectorXr::Zero(Atb_full.size());
	for (int full_i = 0; full_i < guess_full.size(); ++full_i) {
		guess_full[full_i] = solution_small[as_small_index(full_i)];
	}

	return guess_full;
}

/// Break the lattice into tiles, each tile_size^D big.
/// Each tile is solved separately, and the results are combined.
/// The produces a result where the high frequency components are very accurate.
VectorXr tile_solver(
	const SparseMatrix&     AtA_full,
	const VectorXr&         Atb_full,
	const VectorXr&         guess_full,
	const std::vector<int>& sizes_full,
	int                     tile_size)
{
	LOG_SCOPE_F(INFO, "tile_solver");
	CHECK_GE_F(tile_size, 2);

	std::vector<int> num_tiles;
	int num_tiles_total = 1;
	int unknowns_per_tile = 1; // tile_size^D
	for (auto size : sizes_full) {
		num_tiles.push_back((size + tile_size - 1) / tile_size);
		num_tiles_total *= num_tiles.back();
		unknowns_per_tile *= tile_size;
	}

	LOG_F(INFO, "num_tiles_total: %d", num_tiles_total);

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

	for (int full_index = 0; full_index < Atb_full.size(); ++full_index) {
		ERROR_CONTEXT("full_index", full_index);
		int tile_index, index_in_tile;
		std::tie(tile_index, index_in_tile) = calc_tile_and_index(full_index);
		tiles[tile_index].rhs[index_in_tile] = Atb_full[full_index];
	}

	for (int k=0; k < AtA_full.outerSize(); ++k) {
		for (SparseMatrix::InnerIterator it(AtA_full, k); it; ++it) {
			ERROR_CONTEXT("row", it.row());
			ERROR_CONTEXT("col", it.col());
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

	LOG_SCOPE_F(INFO, "solving tiles");
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

std::vector<float> solve_sparse_linear_approximate_lattice(
	const std::vector<Triplet>& triplets,
	const std::vector<float>&   rhs,
	const std::vector<int>&     sizes,
	const SolveOptions&         options)
{
	LOG_SCOPE_F(INFO, "solve_sparse_linear_approximate_lattice");
	int num_unknowns = 1;
	for (auto size : sizes) { num_unknowns *= size; }

	const SparseMatrix A = as_sparse_matrix(triplets, rhs.size(), num_unknowns);
	const SparseMatrix AtA = make_square(A);
	const VectorXr Atb = A.transpose() * as_eigen_vector(rhs);

	LOG_F(INFO, "A nnz:   %lu (%.3f%%)", A.nonZeros(),
	      100.0f * A.nonZeros() / (A.rows() * A.cols()));

	LOG_F(INFO, "AtA nnz: %lu (%.3f%%)", AtA.nonZeros(),
	      100.0f * AtA.nonZeros() / (AtA.rows() * AtA.cols()));
	// ------------------------------------------------------------------------

	VectorXr guess = downscale_solver(AtA, Atb, sizes, options.downscale_factor);

	if (options.tile) {
		guess = tile_solver(AtA, Atb, guess, sizes, options.tile_size);
	}

	if (!options.cg) {
		return as_std_vector(guess);
	}

	LOG_SCOPE_F(INFO, "solveWithGuess");
	// Eigen::BiCGSTAB<SparseMatrix> solver(AtA); // Fails
	// Eigen::LeastSquaresConjugateGradient<SparseMatrix> solver(AtA); // Ringing
	Eigen::ConjugateGradient<SparseMatrix> solver(AtA); // Good
	solver.setTolerance(options.error_tolerance);
	const VectorXr solution = solver.solveWithGuess(Atb, guess);

	LOG_F(INFO, "CG iterations: %lu", solver.iterations());
	LOG_F(INFO, "CG error:      %f",  solver.error());

	if (solver.info() != Eigen::Success) {
		LOG_F(WARNING, "solver.solveWithGuess failed");
		return as_std_vector(guess);
	}

	return as_std_vector(solution);
}
