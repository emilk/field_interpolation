#include "sparse_linear.hpp"

#include <Eigen/Eigen>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>

#include <loguru.hpp>

using VectorXr = Eigen::Matrix<float, Eigen::Dynamic, 1>;

Eigen::SparseMatrix<float> as_sparse_matrix(
    const std::vector<Triplet>& triplets, size_t num_rows, size_t num_columns)
{
	LOG_SCOPE_F(INFO, "as_sparse_matrix");

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

	Eigen::SparseMatrix<float> A(num_rows, num_columns);
	LOG_SCOPE_F(INFO, "setFromTriplets");
	A.setFromTriplets(eigen_triplets.begin(), eigen_triplets.end());
	A.makeCompressed();
	return A;
}

VectorXr as_eigen_vector(const std::vector<float>& values)
{
	return Eigen::Map<VectorXr>(const_cast<float*>(values.data()), values.size());
}

std::vector<float> solve_sparse_linear(
	int                         num_columns,
	const std::vector<Triplet>& triplets,
	const std::vector<float>&   rhs)
{
	LOG_SCOPE_F(INFO, "solve_sparse_linear");
	const auto A = as_sparse_matrix(triplets, rhs.size(), num_columns);

	Eigen::SparseMatrix<float> AtA;

	{
		LOG_SCOPE_F(INFO, "AtA");
		AtA = A.transpose() * A;
		CHECK_EQ_F(AtA.rows(), AtA.cols());
		AtA.makeCompressed();
	}

	VectorXr Atb = A.transpose() * as_eigen_vector(rhs);

	LOG_SCOPE_F(INFO, "Solve");
	Eigen::SparseLU<Eigen::SparseMatrix<float>, Eigen::COLAMDOrdering<int>> solver(AtA);

	if (solver.info() != Eigen::Success) {
		LOG_F(WARNING, "solver.compute failed");
		return {};
	}

	VectorXr solution = solver.solve(Atb);

	if (solver.info() != Eigen::Success) {
		LOG_F(WARNING, "solver.solve failed");
		return {};
	}

	return std::vector<float>(solution.data(), solution.data() + solution.rows() * solution.cols());
}

std::vector<float> solve_sparse_linear_with_guess(
	const std::vector<Triplet>& triplets,
	const std::vector<float>&   rhs,
	const std::vector<float>&   guess)
{
	const auto A = as_sparse_matrix(triplets, rhs.size(), guess.size());

	Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<float>> solver(A);
	const VectorXr solution = solver.solveWithGuess(as_eigen_vector(rhs), as_eigen_vector(guess));
	LOG_F(INFO, "CG iterations: %lu", solver.iterations());
	LOG_F(INFO, "CG error:      %f",  solver.error());

	if (solver.info() != Eigen::Success) {
		LOG_F(WARNING, "solver.solve failed");
		return {};
	}

	return std::vector<float>(solution.data(), solution.data() + solution.rows() * solution.cols());
}
