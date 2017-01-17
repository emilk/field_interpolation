#include "sdf.hpp"

#include <cmath>
#include <ostream>

#include <loguru.hpp>

const int TWO_TO_MAX_DIM = (1 << 4);

std::ostream& operator<<(std::ostream& os, const LinearEquation& eq)
{
	const size_t num_rows = eq.rhs.size();
	std::vector<std::vector<Triplet>> row_triplets(num_rows);

	for (const auto& triplet : eq.triplets) {
		row_triplets[triplet.row].push_back(triplet);
	}

	for (size_t i = 0; i < num_rows; ++i) {
		os << eq.rhs[i] << " = ";
		for (const auto& triplet : row_triplets[i]) {
			os << triplet.value << " * x" << triplet.col << "  +  ";
		}
		os << "\n";
	}
	return os;
}

void add_equation(LinearEquation* eq, float rhs, std::initializer_list<LinearEquationPair> pairs)
{
	// bool all_zero = rhs == 0;
	bool all_zero = true;
	int row = eq->rhs.size();
	for (const auto& pair : pairs) {
		if (pair.value != 0) {
			eq->triplets.emplace_back(row, pair.column, pair.value);
			all_zero = false;
		}
	}
	if (!all_zero) {
		eq->rhs.emplace_back(rhs);
	}
}

/// Computed coefficients for multi-dimensional linear interpolation of 2^D neighbors.
/// Returns false iff this is too close to, or outside of, a border.
bool multilerp(
	int                     out_index[],
	float                   out_weight[],
	const std::vector<int>& sizes,
	const float             in_pos[],
	int                     extra_bound)
{
	int num_dim = sizes.size();
	CHECK_F(1 <= num_dim && num_dim <= MAX_DIM);
	int floored[MAX_DIM];
	float t[MAX_DIM];

	for (int d = 0; d < num_dim; ++d) {
		floored[d] = std::floor(in_pos[d]);
		t[d] = in_pos[d] - floored[d];

		if (floored[d] < 0 || sizes[d] <= floored[d] + 1 + extra_bound) {
			return false;
		}
	}

	for (int i = 0; i < (1 << num_dim); ++i) {
		int index = 0;
		int stride = 1;
		float weight = 1;
		for (int d = 0; d < num_dim; ++d) {
			const int set = (i >> d) & 1;
			int dim_coord = floored[d] + set;
			index  += stride * dim_coord;
			weight *= (set ? t[d] : 1.0f - t[d]);
			stride *= sizes[d];
		}
		out_index[i] = index;
		out_weight[i] = weight;
	}

	return true;
}

bool add_value_constraint(
	LatticeField* field,
	const float   pos[],
	float         value,
	float         strength)
{
	if (strength == 0) { return false; }

	int indices[TWO_TO_MAX_DIM];
	float weights[TWO_TO_MAX_DIM];
	if (!multilerp(indices, weights, field->sizes, pos, 0)) {
		return false;
	}

	#if 0
	// One equation for all four...
	int row = field->eq.rhs.size();
	for (size_t i = 0; i < (1 << field->sizes.size()); ++i) {
		field->eq.triplets.emplace_back(row, indices[i], weights[i] * strength);
	}
	field->eq.rhs.emplace_back(value * strength);
	#else
	// ...or four individual ones?
	for (size_t i = 0; i < (1 << field->sizes.size()); ++i) {
		add_equation(&field->eq, value * strength * weights[i], {
			{indices[i], weights[i] * strength},
		});
	}
	#endif
	return true;
}

bool add_gradient_constraint(
	LatticeField* field,
	const float   pos[],
	const float   gradient[],
	float         strength)
{
	/*
	We spread the contribution using bilinear interpolation.

	Consider the coordinate 3.0 - it should spread the weights equally over neighbors:
		(x[3] - x[2] = dx) * 0.5
		(x[4] - x[3] = dx) * 0.5

	Now consider the coordinate 3.2. It should spread more weight on the next constraint:
		(x[3] - x[2] = dx) * 0.3
		(x[4] - x[3] = dx) * 0.7
	*/

	int num_dim = field->sizes.size();

	float adjusted_pos[MAX_DIM];
	for (int d = 0; d < num_dim; ++d) {
		adjusted_pos[d] = pos[d] - 0.5f;
	}

	int indices[TWO_TO_MAX_DIM];
	float weights[TWO_TO_MAX_DIM];
	if (!multilerp(indices, weights, field->sizes, adjusted_pos, 1)) {
		return false;
	}

	for (size_t i = 0; i < (1 << num_dim); ++i) {
		int stride = 1;
		for (int d = 0; d < num_dim; ++d) {
			// d f(x, y) / dx = gradient[0]
			// d f(x, y) / dy = gradient[1]
			// ...
			add_equation(&field->eq, gradient[d] * strength * weights[i], {
				{indices[i] + 0,      -strength * weights[i]},
				{indices[i] + stride, +strength * weights[i]},
			});
			stride *= field->sizes[d];
		}
	}
	return true;
}

/// Add smoothness constraints between the unknowns:  index - stride, index, index + stride, ...
void add_model_constraint(
	LatticeField*    field,
	const Strengths& strengths,
	int              index,    // index of this value
	int              dim_cord, // coordinate on this dimension
	int              dim_size, // size of this dimension
	int              stride)   // distance between adjacent element along this dimension
{
	if (0 <= dim_cord && dim_cord < dim_size) {
		// f(x) = 0
		// Tikhonov diagonal regularization
		add_equation(&field->eq, 0.0f, {
			{index, strengths.model_0},
		});
	}

	if (0 <= dim_cord && dim_cord + 1 < dim_size) {
		// f′(x) = 0   ⇔   f(x) = f(x + 1)
		add_equation(&field->eq, 0.0f, {
			{index + 0,      -0.5f * strengths.model_1},
			{index + stride, +0.5f * strengths.model_1},
		});
	}

	if (1 <= dim_cord && dim_cord + 1 < dim_size) {
		// f″(x) = 0   ⇔   f′(x - ½) = f′(x + ½)
		add_equation(&field->eq, 0.0f, {
			{index - stride, +0.25f * strengths.model_2},
			{index + 0,      -0.50f * strengths.model_2},
			{index + stride, +0.25f * strengths.model_2},
		});
	}

	if (2 <= dim_cord && dim_cord + 2 < dim_size) {
		// f‴(x) = 0   ⇔   f″(x - 1) = f″(x + 1)
		add_equation(&field->eq, 0.0f, {
			{index - 2 * stride, +1.0f / 6.0f * strengths.model_3},
			{index - 1 * stride, -2.0f / 6.0f * strengths.model_3},
			{index + 1 * stride, +2.0f / 6.0f * strengths.model_3},
			{index + 2 * stride, -1.0f / 6.0f * strengths.model_3},
		});
	}
}

void add_field_constraints(
	LatticeField*    field,
	const Strengths& strengths)
{
	int num_unknowns = 1;
	for (auto dimension_size : field->sizes) {
		num_unknowns *= dimension_size;
	}
	for (int index = 0; index < num_unknowns; ++index) {
		int stride = 1;
		int coordinate = index;
		for (auto dimension_size : field->sizes) {
			int dim_cord = coordinate % dimension_size;
			add_model_constraint(field, strengths, index, dim_cord, dimension_size, stride);
			coordinate /= dimension_size;
			stride *= dimension_size;
		}
	}
}

LatticeField sdf_from_points(
	const std::vector<int>& sizes,
	const Strengths&        strengths,
	const int               num_points,
	const float             positions[],
	const float*            normals,
	const float*            point_weights)
{
	CHECK_NOTNULL_F(positions);

	int num_dim = sizes.size();
	LatticeField field{sizes};

	add_field_constraints(&field, strengths);
	for (int i = 0; i < num_points; ++i) {
		float weight = point_weights ? point_weights[i] : 1.0f;
		const float* pos = positions + i * num_dim;
		add_value_constraint(&field, pos, 0.0f, weight * strengths.data_pos);
		if (normals) {
			add_gradient_constraint(&field, pos, normals + i * num_dim, weight * strengths.data_gradient);
		}
	}

	return field;
}
