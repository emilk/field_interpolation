#include "sparse_linear.hpp"

#include <Eigen/Eigen>
#include <Eigen/Sparse>

#include <loguru.hpp>

using VectorXr = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

std::vector<Real> solve_sparse_linear(
	int                               columns,
	const std::vector<Triplet<Real>>& triplets,
	const std::vector<Real>&          rhs)
{
	std::vector<Eigen::Triplet<Real>> eigen_triplets;
	eigen_triplets.reserve(triplets.size());
	for (const auto& triplet : triplets) {
		CHECK_GE_F(triplet.col, 0);
		CHECK_GE_F(triplet.row, 0);
		CHECK_LT_F(triplet.col, columns);
		CHECK_LT_F(triplet.row, rhs.size());
		if (triplet.value != 0.0f) {
			eigen_triplets.emplace_back(triplet.row, triplet.col, triplet.value);
		}
	}

	Eigen::SparseMatrix<Real> A(rhs.size(), columns);
	A.setFromTriplets(eigen_triplets.begin(), eigen_triplets.end());
	A.makeCompressed();

	const VectorXr b = Eigen::Map<VectorXr>(const_cast<Real*>(rhs.data()), rhs.size());

	Eigen::SparseMatrix<Real> AtA = A.transpose() * A;
	CHECK_EQ_F(AtA.rows(), AtA.cols());
	AtA.makeCompressed();

	Eigen::SparseLU<Eigen::SparseMatrix<Real>, Eigen::COLAMDOrdering<int>> solver;

	solver.compute(AtA);
	if (solver.info() != Eigen::Success) {
		LOG_F(WARNING, "solver.compute failed");
		return {};
	}

	VectorXr Atb = A.transpose() * b;
	VectorXr solution = solver.solve(Atb);

	if(solver.info() != Eigen::Success) {
		LOG_F(WARNING, "solver.solve failed");
		return {};
	}

	return std::vector<Real>(solution.data(), solution.data() + solution.rows() * solution.cols());
}
