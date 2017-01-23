// By Emil Ernerfeldt 2017
// LICENSE:
//   This software is dual-licensed to the public domain and under the following
//   license: you are granted a perpetual, irrevocable license to copy, modify,
//   publish, and distribute this file as you see fit.

#include "dual_contouring_2d.hpp"

#include <loguru.hpp>

namespace dc {

using Real = float;

const size_t kNumCorners     = 4;
const Real   kRegularization = 0.001f;
const Index  kInvalidIndex   = static_cast<Index>(-1);

Vec2 solve_lin_eq_2d(const Real A[4], const Real b[2])
{
	const Real det = A[0] * A[3] - A[1] * A[2];

	return Vec2{
		static_cast<float>((b[0] * A[3] - A[1] * b[1]) / det),
		static_cast<float>((b[1] * A[0] - A[2] * b[0]) / det),
	};
}

Vec2 least_squares_2d(const Real* A, const Real* b, size_t num_rows)
{
	Real AtA[4] = { 0.0, 0.0, 0.0, 0.0 };
	Real Atb[2] = { 0.0, 0.0 };

	for (size_t i = 0; i < num_rows; ++i) {
		AtA[0] += A[2 * i + 0] * A[2 * i + 0];
		AtA[1] += A[2 * i + 0] * A[2 * i + 1];
		AtA[2] += A[2 * i + 1] * A[2 * i + 0];
		AtA[3] += A[2 * i + 1] * A[2 * i + 1];

		Atb[0] += A[2 * i + 0] * b[i];
		Atb[1] += A[2 * i + 1] * b[i];
	}

	return solve_lin_eq_2d(AtA, Atb);
}

void dual_contouring_2d(
    std::vector<Vec2>* out_vertices, std::vector<Index>* out_line_segments,
    size_t width, size_t height, const float* distances, const Vec2* gradients)
{
	const auto index = [=](size_t x, size_t y) { return y * width + x; };

	std::vector<size_t> cell_vertex_indices(width * height, kInvalidIndex);

	for (size_t y = 0; y + 1 < height; ++y) {
		for (size_t x = 0; x + 1 < width; ++x) {
			size_t num_inside = 0;
			bool corner_inside[kNumCorners];
			for (size_t ci = 0; ci < kNumCorners; ++ci) {
				bool is_inside = distances[index(x + ci % 2, y + ci / 2)] <= 0;
				corner_inside[ci] = is_inside;
				num_inside += is_inside;
			}
			if (num_inside == 0 || num_inside == kNumCorners) {
				// No crossings in this cell
				continue;
			}

			// All corners connected to an edge with a sign change:
			bool corners_with_crossing[kNumCorners] = {
				corner_inside[0b00] != corner_inside[0b01] || corner_inside[0b00] != corner_inside[0b10],
				corner_inside[0b01] != corner_inside[0b00] || corner_inside[0b01] != corner_inside[0b11],
				corner_inside[0b10] != corner_inside[0b11] || corner_inside[0b10] != corner_inside[0b00],
				corner_inside[0b11] != corner_inside[0b10] || corner_inside[0b11] != corner_inside[0b01],
			};

			// Construct the linear equation Ax = b, where x is where (in cell coordinates) to put the vertex:
			const size_t kNumEquations = kNumCorners + 2;
			Real A[kNumEquations * 2] = {0};
			Real b[kNumEquations] = {0};

			for (size_t ci = 0; ci < kNumCorners; ++ci) {
				if (!corners_with_crossing[ci]) { continue; }
				const size_t dx = ci % 2;
				const size_t dy = ci / 2;
				const size_t ix = index(x + dx, y + dy);
				const Real distance = distances[ix];
				const Real gradient_x = gradients[ix].x;
				const Real gradient_y = gradients[ix].y;
				A[2 * ci + 0] = gradient_x;
				A[2 * ci + 1] = gradient_y;
				b[ci] =
					static_cast<Real>(dx) * gradient_x +
					static_cast<Real>(dy) * gradient_y
					- distance;
			}

			// Instead of clamping, increase regularization until vertex is withing the cell:
			Vec2 vertex;
			Real regularization = kRegularization;
			do {
				// Add a weak push towards center of the cell as a regularization:
				A[4 * 2 + 0] = regularization;
				A[4 * 2 + 1] = 0;
				b[4] = 0.5f * regularization;
				A[5 * 2 + 0] = 0;
				A[5 * 2 + 1] = regularization;
				b[5] = 0.5f * regularization;

				vertex = least_squares_2d(A, b, kNumEquations);
				regularization *= 2.0f;
			} while (vertex.x < 0 || 1 < vertex.x || vertex.y < 0 || 1 < vertex.y);

			cell_vertex_indices[index(x, y)] = out_vertices->size();
			out_vertices->push_back(Vec2{x + vertex.x, y + vertex.y});
		}
	}

	// ------------------------------------------------------------------------
	// Now we have the vertices. Time to collect edges:

	for (size_t y = 0; y + 1 < height; ++y) {
		for (size_t x = 0; x + 1 < width; ++x) {
			auto vert_idx = cell_vertex_indices[index(x, y)];
			if (vert_idx == kInvalidIndex) { continue; }

			bool corner_inside[kNumCorners];
			for (size_t ci = 0; ci < kNumCorners; ++ci) {
				corner_inside[ci] = distances[index(x + ci % 2, y + ci / 2)] <= 0;
			}

			if (corner_inside[0b01] != corner_inside[0b11]) {
				// Connect to neighbor to our right:
				auto neighbor_vert_idx = cell_vertex_indices[index(x + 1, y)];
				if (neighbor_vert_idx != kInvalidIndex) {
					if (corner_inside[0b11]) {
						out_line_segments->push_back(vert_idx);
						out_line_segments->push_back(neighbor_vert_idx);
					} else {
						out_line_segments->push_back(neighbor_vert_idx);
						out_line_segments->push_back(vert_idx);
					}
				}
			}

			if (corner_inside[0b10] != corner_inside[0b11]) {
				// Connect to neighbor below us:
				auto neighbor_vert_idx = cell_vertex_indices[index(x, y + 1)];
				if (neighbor_vert_idx != kInvalidIndex) {
					if (corner_inside[0b11]) {
						out_line_segments->push_back(neighbor_vert_idx);
						out_line_segments->push_back(vert_idx);
					} else {
						out_line_segments->push_back(vert_idx);
						out_line_segments->push_back(neighbor_vert_idx);
					}
				}
			}
		}
	}
}

void calculate_gradients(Vec2* out_gradients, size_t width, size_t height, const float* distances)
{
	const auto index = [=](size_t x, size_t y) { return y * width + x; };

	for (size_t y = 0; y < height; ++y) {
		for (size_t x = 0; x < width; ++x) {
			Vec2& gradient = out_gradients[index(x, y)];

			if (x == 0) {
				gradient.x = distances[index(x + 1, y)] - distances[index(x, y)];
			} else if (x == width - 1) {
				gradient.x = distances[index(x, y)] - distances[index(x - 1, y)];
			} else {
				gradient.x = (distances[index(x + 1, y)] - distances[index(x - 1, y)]) / 2;
			}

			if (y == 0) {
				gradient.y = distances[index(x, y + 1)] - distances[index(x, y)];
			} else if (y == width - 1) {
				gradient.y = distances[index(x, y)] - distances[index(x, y - 1)];
			} else {
				gradient.y = (distances[index(x, y + 1)] - distances[index(x, y - 1)]) / 2;
			}
		}
	}
}

} // namespace dc
