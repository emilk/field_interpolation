#include "sdf.hpp"

#include <cmath>

#include <loguru.hpp>

const int TWO_TO_MAX_DIM = (1 << 4);

/// Helper to add a row to the linear equation.
void add_equation(LinearEquation* eq, float rhs, std::initializer_list<LinearEquationPair> pairs)
{
	bool all_zero = rhs == 0;
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
	int         out_index[],
	float       out_weight[],
	int         num_dim,
	const int   sizes[],
	const float in_pos[],
	int         extra_bound)
{
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
	int indices[TWO_TO_MAX_DIM];
	float weights[TWO_TO_MAX_DIM];
	if (!multilerp(indices, weights, field->num_dim, field->sizes, pos, 0)) {
		return false;
	}

	for (size_t i = 0; i < (1 << field->num_dim); ++i) {
		add_equation(&field->eq, value * strength, {
			{indices[i], weights[i] * strength},
		});
	}
	return true;
}

bool add_gradient_constraint(
	LatticeField* field,
	const float   pos[],
	const float   gradient[],
	float         strength)
{
	/*
	We want to spread the contribution using bilinear interpolation:

	Consider the coordinate 3 - it should spread the weights equally over neighbors:
		(x[3] - x[2] = dx) * 0.5
		(x[4] - x[3] = dx) * 0.5

	Now Consider the coordinate 3.2. It should spread more weight on the next constraint:
		(x[3] - x[2] = dx) * 0.3
		(x[4] - x[3] = dx) * 0.7
	*/

	float adjusted_pos[MAX_DIM];
	for (int d = 0; d < field->num_dim; ++d) {
		adjusted_pos[d] = pos[d] - 0.5f;
	}

	int indices[TWO_TO_MAX_DIM];
	float weights[TWO_TO_MAX_DIM];
	if (!multilerp(indices, weights, field->num_dim, field->sizes, adjusted_pos, 1)) {
		return false;
	}

	for (size_t i = 0; i < (1 << field->num_dim); ++i) {
		int stride = 1;
		for (int d = 0; d < field->num_dim; ++d) {
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
		add_equation(&field->eq, 0.0f, {
			{index, strengths.model_0},
		});
	}

	if (0 <= dim_cord && dim_cord + 1 < dim_size) {
		// f′(x) = 0   ⇔   f(x) = f(x + 1)
		add_equation(&field->eq, 0.0f, {
			{index + 0,      +strengths.model_1},
			{index + stride, +strengths.model_1},
		});
	}

	if (1 <= dim_cord && dim_cord + 1 < dim_size) {
		// f″(x) = 0   ⇔   f′(x - ½) = f′(x + ½)
		add_equation(&field->eq, 0.0f, {
			{index - stride, +1 * strengths.model_2},
			{index + 0,      -2 * strengths.model_2},
			{index + stride, +1 * strengths.model_2},
		});
	}

	if (2 <= dim_cord && dim_cord + 2 < dim_size) {
		// f‴(x) = 0   ⇔   f″(x - 1) = f″(x + 1)
		add_equation(&field->eq, 0.0f, {
			{index - 2 * stride, +1 * strengths.model_3},
			{index - 1 * stride, -2 * strengths.model_3},
			{index + 1 * stride, +2 * strengths.model_3},
			{index + 2 * stride, -1 * strengths.model_3},
		});
	}
}

void add_field_constraints(
	LatticeField*    field,
	const Strengths& strengths)
{
	int num_unknowns = 1;
	for (int d = 0; d < field->num_dim; ++d) {
		num_unknowns *= field->sizes[d];
	}
	for (int index = 0; index < num_unknowns; ++index) {
		int stride = 1;
		int coordinate = index;
		for (int d = 0; d < field->num_dim; ++d) {
			int dim_cord = coordinate % field->sizes[d];
			add_model_constraint(field, strengths, index, dim_cord, field->sizes[d], stride);
			coordinate /= field->sizes[d];
			stride *= field->sizes[d];
		}
	}
}

LatticeField sdf_from_points(
    int              num_dim,
    const int        sizes[],
    const Strengths& strengths,
    int              num_points,
    const float      positions[],
    const float*     normals,
    const float*     point_weights)
{
	CHECK_NOTNULL_F(positions);

	LatticeField field(num_dim, sizes);

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
