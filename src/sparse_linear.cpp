#include "sparse_linear.hpp"

#include <Eigen/Eigen>
#include <Eigen/Sparse>

#include <loguru.hpp>

template<typename T>
std::vector<T> solve_sparse_linear_t(
	int                         columns,
	const std::vector<Triplet>& triplets,
	const std::vector<T>&       rhs)
{
	using VectorXr = Eigen::Matrix<T, Eigen::Dynamic, 1>;

	std::vector<Eigen::Triplet<T>> eigen_triplets;
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

	Eigen::SparseMatrix<T> A(rhs.size(), columns);
	A.setFromTriplets(eigen_triplets.begin(), eigen_triplets.end());
	A.makeCompressed();

	Eigen::SparseMatrix<T> AtA = A.transpose() * A;
	CHECK_EQ_F(AtA.rows(), AtA.cols());
	AtA.makeCompressed();

	Eigen::SparseLU<Eigen::SparseMatrix<T>, Eigen::COLAMDOrdering<int>> solver;

	solver.compute(AtA);
	if (solver.info() != Eigen::Success) {
		LOG_F(WARNING, "solver.compute failed");
		return {};
	}

	const VectorXr b = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>(const_cast<T*>(rhs.data()), rhs.size());
	VectorXr Atb = A.transpose() * b;
	VectorXr solution = solver.solve(Atb);

	if(solver.info() != Eigen::Success) {
		LOG_F(WARNING, "solver.solve failed");
		return {};
	}

	return std::vector<T>(solution.data(), solution.data() + solution.rows() * solution.cols());
}

std::vector<float> solve_sparse_linear(
	int                         columns,
	const std::vector<Triplet>& triplets,
	const std::vector<float>&   rhs,
	bool                        double_precision)
{
	if (double_precision) {
		std::vector<double> rhs_doubles;
		rhs_doubles.reserve(rhs.size());
		for (auto f : rhs) { rhs_doubles.push_back(f); }

		const auto answer_doubles = solve_sparse_linear_t<double>(columns, triplets, rhs_doubles);
		std::vector<float> answer_floats;
		answer_floats.reserve(answer_doubles.size());
		for (auto d : answer_doubles) { answer_floats.push_back(d); }
		return answer_floats;
	} else {
		return solve_sparse_linear_t<float>(columns, triplets, rhs);
	}
}
