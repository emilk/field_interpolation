#include "sdf.hpp"

#include <cmath>

#include <loguru.hpp>

void add_point_constraint(
	std::vector<TripletR>* triplets,
	std::vector<Real>*          rhs,
	size_t                      resolution,
	const Strengths&            strengths,
	const Point&                point)
{
	CHECK_NOTNULL_F(triplets);
	CHECK_NOTNULL_F(rhs);
	ERROR_CONTEXT("row", rhs->size());

	auto index = [=](int x, int y) { return y * resolution + x; };

	const float xf = point.x * resolution;
	const float yf = point.y * resolution;
	const int x_floored = std::floor(xf);
	const int y_floored = std::floor(yf);

	if (x_floored < 0 || resolution <= x_floored) { return; }
	if (y_floored < 0 || resolution <= y_floored) { return; }

	if (0.0f < strengths.data_pos) {
		// Bilinear interpolation of points influence:
		const float tx = xf - x_floored;
		const float ty = yf - y_floored;

		triplets->push_back(TripletR(rhs->size(), index(x_floored, y_floored), strengths.data_pos * (1 - tx) * (1 - ty)));
		rhs->push_back(0.0f);

		if (x_floored + 1 < resolution) {
			triplets->push_back(TripletR(rhs->size(), index(x_floored + 1, y_floored), strengths.data_pos * tx * (1 - ty)));
			rhs->push_back(0.0f);
		}

		if (y_floored + 1 < resolution) {
			triplets->push_back(TripletR(rhs->size(), index(x_floored, y_floored + 1), strengths.data_pos * (1 - tx) * ty));
			rhs->push_back(0.0f);
		}

		if (x_floored + 1 < resolution && y_floored + 1 < resolution) {
			triplets->push_back(TripletR(rhs->size(), index(x_floored + 1, y_floored + 1), strengths.data_pos * tx * ty));
			rhs->push_back(0.0f);
		}
	}

	if (0.0f < strengths.data_normal) {
		// Influence delta between the two closest cells on each axis:
		const int x_rounded = std::round(xf);
		const int y_rounded = std::round(yf);

		if (0 <= x_floored && x_floored + 1 < resolution
		 && 0 <= y_rounded && y_rounded < resolution) {
			// d f(x, y) / dx = point.dx
			triplets->push_back(TripletR(rhs->size(), index(x_floored + 0, y_rounded), -strengths.data_normal));
			triplets->push_back(TripletR(rhs->size(), index(x_floored + 1, y_rounded), +strengths.data_normal));
			rhs->push_back(point.dx * strengths.data_normal);
		}

		if (0 <= x_rounded && x_rounded < resolution
		 && 0 <= y_floored && y_floored + 1 < resolution) {
			// d f(x, y) / dy = point.dy
			triplets->push_back(TripletR(rhs->size(), index(x_rounded, y_floored + 0), -strengths.data_normal));
			triplets->push_back(TripletR(rhs->size(), index(x_rounded, y_floored + 1), +strengths.data_normal));
			rhs->push_back(point.dy * strengths.data_normal);
		}
	}
}

void add_model_constraint(
	std::vector<TripletR>* triplets,
	std::vector<Real>*     rhs,
	size_t                 resolution,
	const Strengths&       strengths,
	int                    x,
	int                    y)
{
	auto index = [=](int x, int y) { return y * resolution + x; };

	if (strengths.model_0 > 0.0f) {
		triplets->push_back(TripletR(rhs->size(), index(x, y), strengths.model_0));
		rhs->push_back(0);
	}

	if (strengths.model_1 > 0.0f) {
		if (0 < x) {
			// df/dx = 0
			triplets->push_back(TripletR(rhs->size(), index(x - 1, y), -strengths.model_1));
			triplets->push_back(TripletR(rhs->size(), index(x + 0, y), +strengths.model_1));
			rhs->push_back(0);
		}

		if (0 < y) {
			// df/dy = 0
			triplets->push_back(TripletR(rhs->size(), index(x, y - 1), -strengths.model_1));
			triplets->push_back(TripletR(rhs->size(), index(x, y + 0), +strengths.model_1));
			rhs->push_back(0);
		}
	}

	if (strengths.model_2 > 0.0f) {
		if (0 < x && x + 1 < resolution) {
			// d2f/dx2 = 0
			triplets->push_back(TripletR(rhs->size(), index(x - 1, y), +1 * strengths.model_2));
			triplets->push_back(TripletR(rhs->size(), index(x + 0, y), -2 * strengths.model_2));
			triplets->push_back(TripletR(rhs->size(), index(x + 1, y), +1 * strengths.model_2));
			rhs->push_back(0);
		}

		if (0 < y && y + 1 < resolution) {
			// d2f/dy2 = 0
			triplets->push_back(TripletR(rhs->size(), index(x, y - 1), +1 * strengths.model_2));
			triplets->push_back(TripletR(rhs->size(), index(x, y + 0), -2 * strengths.model_2));
			triplets->push_back(TripletR(rhs->size(), index(x, y + 1), +1 * strengths.model_2));
			rhs->push_back(0);
		}
	}
}

std::vector<float> generate_sdf(
	size_t resolution, const std::vector<Point>& points, const Strengths& strengths)
{
	ERROR_CONTEXT("resolution", resolution);
	ERROR_CONTEXT("points", points.size());
	LOG_SCOPE_F(INFO, "generate_sdf");
	std::vector<TripletR> triplets;
	std::vector<Real> rhs;

	// Data constraints:
	for (const auto& point : points) {
		add_point_constraint(&triplets, &rhs, resolution, strengths, point);
	}

	// Model constraints:
	for (int y = 0; y < resolution; ++y) {
		for (int x = 0; x < resolution; ++x) {
			add_model_constraint(&triplets, &rhs, resolution, strengths, x, y);
		}
	}

	LOG_F(INFO, "%lu equations", rhs.size());
	LOG_F(INFO, "%lu values in matrix", triplets.size());
	const std::vector<Real> sdf = [&](){
		LOG_SCOPE_F(INFO, "solve_sparse_linear");
		return solve_sparse_linear(resolution * resolution, triplets, rhs);
	}();

	std::vector<float> floats;
	for (auto dist : sdf) {
		floats.push_back(dist);
	}

	return floats;
}
