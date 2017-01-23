// By Emil Ernerfeldt 2017
// LICENSE:
//   This software is dual-licensed to the public domain and under the following
//   license: you are granted a perpetual, irrevocable license to copy, modify,
//   publish, and distribute this file as you see fit.

#pragma once

#include <vector>

namespace dc {

using Index = unsigned;

struct Vec2 { float x, y; };
static_assert(sizeof(Vec2) == sizeof(float) * 2, "Pack");

/**
 * @brief Dual contouring of a iso field on a lattice.
 * @details Will generate line segments for where the iso-field crosses zero.
 *
 * @param out_vertices Vertices will be put here
 * @param out_line_segments Indices into vertices will be put here, in pairs.
 * @param width Width of the lattice
 * @param height Height of the lattice
 * @param distances width * height signed distance field. <= 0 is inside, >0 is outside.
 * @param gradients Local gradient of the distance field.
 */
void dual_contouring_2d(
    std::vector<Vec2>* out_vertices, std::vector<unsigned>* out_line_segments,
    size_t width, size_t height, const float* distances, const Vec2* gradients);

/// Calculated the gradient of a 2D field.
void calculate_gradients(Vec2* out_gradients, size_t width, size_t height, const float* distances);

} // namespace dc
