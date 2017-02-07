#include "field_interpolation.hpp"

#include <cmath>
#include <ostream>

#include <loguru.hpp>

namespace field_interpolation {

const int TWO_TO_MAX_DIM = (1 << 4);

/// Computed coefficients for multi-dimensional linear interpolation of 2^D neighbors.
/// Returns the number of indices to sample from.
/// The indices are put in out_indices, the kernel (interpolation weights) in out_kernel.
int multilerp(
	int                     out_indices[],
	float                   out_kernel[],
	const std::vector<int>& sizes,
	const std::vector<int>& strides,
	const float             in_pos[],
	int                     extra_bound)
{
	const int num_dim = sizes.size();
	CHECK_F(1 <= num_dim && num_dim <= MAX_DIM);

	int floored[MAX_DIM];
	float t[MAX_DIM];

	for (int d = 0; d < num_dim; ++d) {
		floored[d] = std::floor(in_pos[d]);
		t[d] = in_pos[d] - floored[d];
	}

	int num_samples = 0;

	for (int i = 0; i < (1 << num_dim); ++i) {
		int index = 0;
		float weight = 1;
		bool inside = true;
		for (int d = 0; d < num_dim; ++d) {
			const int set = (i >> d) & 1;
			int dim_coord = floored[d] + set;
			index  += strides[d] * dim_coord;
			weight *= (set ? t[d] : 1.0f - t[d]);
			inside &= (0 <= dim_coord && dim_coord + extra_bound < sizes[d]);
		}
		if (inside) {
			out_indices[num_samples] = index;
			out_kernel[num_samples] = weight;
			num_samples += 1;
		}
	}

	return num_samples;
}

bool add_value_constraint(
	LatticeField* field,
	const float   pos[],
	float         value,
	float         constraint_weight)
{
	if (constraint_weight == 0) { return false; }

	int interpolation_indices[TWO_TO_MAX_DIM];
	float interpolation_kernel[TWO_TO_MAX_DIM];
	int num_samples = multilerp(interpolation_indices, interpolation_kernel, field->sizes, field->strides, pos, 0);
	if (num_samples == 0) { return false; }

	int row = field->eq.rhs.size();
	float weight_sum = 0;
	for (int i = 0; i < num_samples; ++i) {
		float sample_weight = interpolation_kernel[i] * constraint_weight;
		field->eq.triplets.emplace_back(row, interpolation_indices[i], sample_weight);
		weight_sum += sample_weight;
	}
	field->eq.rhs.emplace_back(weight_sum * value);

	return true;
}

bool add_value_constraint_nearest_neighbor(
	LatticeField* field,
	const float   pos[],
	const float   gradient[],
	float         value,
	float         weight)
{
	int num_dim = field->num_dim();

	int nearest_index = 0;
	float dist_along_gradient = 0;

	for (int d = 0; d < num_dim; ++d) {
		int nearest_d = std::round(pos[d]);
		if (nearest_d < 0 || field->sizes[d] <= nearest_d) {
			return false;
		}
		dist_along_gradient += (pos[d] - nearest_d) * gradient[d];
		nearest_index += nearest_d * field->strides[d];
	}

	add_equation(&field->eq, Weight{weight}, {value - dist_along_gradient}, {
		{nearest_index, 1.0f}
	});
}

/// Return -1 on out-of-bounds
int cell_index(const LatticeField& field, const float pos[])
{
	int index = 0;
	const int num_dim = field.num_dim();
	for (int d = 0; d < num_dim; ++d) {
		int pos_d = std::floor(pos[d]);
		bool in_lattice = 0 <= pos_d && pos_d + 1 < field.sizes[d];
		if (!in_lattice) { return -1; }
		index += pos_d * field.strides[d];
	}
	return index;
}

bool add_gradient_constraint(
	LatticeField*  field,
	const float    pos[],
	const float    gradient[],
	float          constraint_weight,
	GradientKernel kernel)
{
	if (constraint_weight == 0) { return false; }

	// TODO: add three equations like in http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.440.3739&rep=rep1&type=pdf

	if (kernel == GradientKernel::kNearestNeighbor) {
		int index = cell_index(*field, pos);
		if (index < 0) { return false; }

		const int num_dim = field->sizes.size();

		for (int d = 0; d < num_dim; ++d) {
			// d f(x, y) / dx = gradient[0]
			// d f(x, y) / dy = gradient[1]
			// ...
			add_equation(&field->eq, Weight{constraint_weight}, Rhs{gradient[d]}, {
				{index + 0,                 -1.0f},
				{index + field->strides[d], +1.0f},
			});
		}
		return true;
	} else if (kernel == GradientKernel::kCellEdges) {
		/*
		This method was described in SSD: Smooth Signed Distance Surface Reconstruction
		http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.440.3739&rep=rep1&type=pdf

		Find the voxel cell containing the point. The voxel has the corners A, B, C, D:
			A B
			C D
		Add constraints:
			((A - B) + (D - C)) / 2 = dx
			((C - A) + (D - B)) / 2 = dy

		So this will add num_dim equations with 2^num_dim terms in each.
		*/

		int index = cell_index(*field, pos);
		if (index < 0) { return false; }

		const int num_dim = field->sizes.size();

		for (int d = 0; d < num_dim; ++d) {
			const int row = field->eq.rhs.size();
			const int num_corners = (1 << num_dim);
			const float term_weight = constraint_weight * 2.0f / num_corners;

			for (int corner = 0; corner < num_corners; ++corner) {
				int corner_index = index;
				for (int oa = 0; oa < num_dim; ++oa) {
					int is_along_oa = (corner >> oa) % 2;
					corner_index += field->strides[oa] * is_along_oa;
				}
				bool is_along_d = (corner >> d) % 2;
				float sign = is_along_d ? +1.0f : -1.0f;
				field->eq.triplets.emplace_back(row, corner_index, sign * term_weight);
			}
			field->eq.rhs.emplace_back(constraint_weight * gradient[d]);
		}
		return true;
	} else if (kernel == GradientKernel::kLinearInterpolation) {
		/*
		We spread the contribution using bilinear interpolation.

		Case A):
			pos = 3.5: put all weight onto one equation:
				(x[4] - x[3] = dx) * 1.0

		Case B):
			pos = 3.0: spread the weights equally over two neighbors:
				(x[3] - x[2] = dx) * 0.5
				(x[4] - x[3] = dx) * 0.5

		Case C):
			pos = 3.25: spread more weight on the next constraint:
				(x[3] - x[2] = dx) * 0.25
				(x[4] - x[3] = dx) * 0.75

		We combine these constraints into one equation.
		*/

		const int num_dim = field->sizes.size();

		float adjusted_pos[MAX_DIM];
		for (int d = 0; d < num_dim; ++d) {
			adjusted_pos[d] = pos[d] - 0.5f;
		}

		int interpolation_indices[TWO_TO_MAX_DIM];
		float interpolation_kernel[TWO_TO_MAX_DIM];
		int num_samples = multilerp(interpolation_indices, interpolation_kernel, field->sizes, field->strides, adjusted_pos, 1);
		if (num_samples == 0) { return false; }

		for (int d = 0; d < num_dim; ++d) {
			int row = field->eq.rhs.size();
			float weight_sum = 0;
			for (int i = 0; i < num_samples; ++i) {
				// d f(x, y) / dx = gradient[0]
				// d f(x, y) / dy = gradient[1]
				// ...
				const float sample_weight = interpolation_kernel[i] * constraint_weight;
				field->eq.triplets.emplace_back(row, interpolation_indices[i] + 0,                 -sample_weight);
				field->eq.triplets.emplace_back(row, interpolation_indices[i] + field->strides[d], +sample_weight);
				weight_sum += sample_weight;
			}
			field->eq.rhs.emplace_back(weight_sum * gradient[d]);
		}

		return true;
	} else {
		ABORT_F("Unknown gradient kernel: %d", static_cast<int>(kernel));
	}
}

/// Add smoothness constraints at the given coordinate along the given dimension
void add_model_constraint(
	LatticeField*  field,
	const Weights& weights,
	const int      coordinate[MAX_DIM],
	int            index,    // index of this value
	int            d)        // dimension
{
	const int size     = field->sizes[d];
	const int stride   = field->strides[d];
	const int dim_cord = coordinate[d];

	// These weights come from Pascal's triangle.
	// See also https://en.wikipedia.org/wiki/Finite_difference_coefficient

	if (weights.model_0 > 0 && 0 <= dim_cord && dim_cord < size) {
		// f(x) = 0
		// Tikhonov diagonal regularization
		add_equation(&field->eq, Weight{weights.model_0}, Rhs{0.0f}, {
			{index, 1.0f},
		});
	}

	if (weights.model_1 > 0 && 0 <= dim_cord && dim_cord + 1 < size) {
		// f′(x) = 0   ⇔   f(x) = f(x + 1)
		add_equation(&field->eq, Weight{weights.model_1}, Rhs{0.0f}, {
			{index + 0 * stride, -1.0f},
			{index + 1 * stride, +1.0f},
		});
	}

	if (weights.model_2 > 0 && 0 <= dim_cord && dim_cord + 2 < size) {
		// f″(x) = 0   ⇔   f′(x - ½) = f′(x + ½)
		add_equation(&field->eq, Weight{weights.model_2}, Rhs{0.0f}, {
			{index + 0 * stride, +1.0f},
			{index + 1 * stride, -2.0f},
			{index + 2 * stride, +1.0f},
		});
	}

	if (weights.model_3 > 0 && 0 <= dim_cord && dim_cord + 3 < size) {
		// f‴(x) = 0   ⇔   f″(x - ½) = f″(x + ½)
		add_equation(&field->eq, Weight{weights.model_3}, Rhs{0.0f}, {
			{index + 0 * stride, +1.0f},
			{index + 1 * stride, -3.0f},
			{index + 2 * stride, +3.0f},
			{index + 3 * stride, -1.0f},
		});
	}

	if (weights.model_4 > 0 && 0 <= dim_cord && dim_cord + 4 < size) {
		// f⁗(x) = 0   ⇔   f‴(x - ½) = f‴(x + ½)
		add_equation(&field->eq, Weight{weights.model_4}, Rhs{0.0f}, {
			{index + 0 * stride, +1.0f},
			{index + 1 * stride, -4.0f},
			{index + 2 * stride, +6.0f},
			{index + 3 * stride, -4.0f},
			{index + 4 * stride, +1.0f},
		});
	}

	if (weights.gradient_smoothness > 0 && 0 <= dim_cord && dim_cord + 1 < size) {
		// The gradient along d should be equal in two neighboring edges:
		for (int orthogonal_dim = 0; orthogonal_dim < field->sizes.size(); ++orthogonal_dim) {
			if (d == orthogonal_dim) { continue; }
			if (coordinate[orthogonal_dim] + 1 >= field->sizes[orthogonal_dim]) { continue; }
			add_equation(&field->eq, Weight{weights.gradient_smoothness}, Rhs{0.0f}, {
				{index + 0 * field->strides[orthogonal_dim] + 0 * field->strides[d], -1.0f},
				{index + 0 * field->strides[orthogonal_dim] + 1 * field->strides[d], +1.0f},
				{index + 1 * field->strides[orthogonal_dim] + 0 * field->strides[d], +1.0f},
				{index + 1 * field->strides[orthogonal_dim] + 1 * field->strides[d], -1.0f},
			});
		}
	}
}

void coordinate_from_index(const std::vector<int>& sizes, int coordinate[MAX_DIM], int index)
{
	for (int d = 0; d < sizes.size(); ++d) {
		coordinate[d] = index % sizes[d];
		index /= sizes[d];
	}
}

void add_field_constraints(
	LatticeField*  field,
	const Weights& weights)
{
	int num_unknowns = 1;
	for (auto dimension_size : field->sizes) {
		num_unknowns *= dimension_size;
	}
	for (int index = 0; index < num_unknowns; ++index) {
		int coordinate[MAX_DIM];
		coordinate_from_index(field->sizes, coordinate, index);
		for (int d = 0; d < field->sizes.size(); ++d) {
			add_model_constraint(field, weights, coordinate, index, d);
		}
	}
}

LatticeField sdf_from_points(
	const std::vector<int>& sizes,
	const Weights&          weights,
	const int               num_points,
	const float             positions[],
	const float*            normals,
	const float*            point_weights)
{
	LOG_SCOPE_F(1, "sdf_from_points");
	CHECK_NOTNULL_F(positions);

	int num_dim = sizes.size();
	LatticeField field{sizes};

	add_field_constraints(&field, weights);

	for (int i = 0; i < num_points; ++i) {
		float weight = point_weights ? point_weights[i] : 1.0f;
		const float* pos = positions + i * num_dim;
		const float* gradient = normals + i * num_dim;
		if (weights.value_kernel == ValueKernel::kNearestNeighbor) {
			CHECK_NOTNULL_F(normals);
			add_value_constraint_nearest_neighbor(&field, pos, gradient, 0.0f, weight * weights.data_pos);
		} else {
			add_value_constraint(&field, pos, 0.0f, weight * weights.data_pos);
		}

		if (normals) {
			add_gradient_constraint(&field, pos, gradient, weight * weights.data_gradient, weights.gradient_kernel);
		}
	}

	return field;
}

std::vector<float> generate_error_map(
	const std::vector<Triplet>& triplets,
	const std::vector<float>&   solution,
	const std::vector<float>&   rhs)
{
	std::vector<float> row_errors = rhs;
	std::vector<float> sum_of_value_sq(rhs.size(), 0.0f);

	for (const auto& triplet : triplets) {
		row_errors[triplet.row] -= solution[triplet.col] * triplet.value;
		sum_of_value_sq[triplet.row] += triplet.value * triplet.value;
	}

	for (auto& error : row_errors) {
		error *= error;
	}

	std::vector<float> heatmap(solution.size(), 0.0f);

	for (const auto& triplet : triplets) {
		if (sum_of_value_sq[triplet.row] != 0) {
			float blame_fraction = (triplet.value * triplet.value) / sum_of_value_sq[triplet.row];
			heatmap[triplet.col] += blame_fraction * row_errors[triplet.row];
		}
	}

	return heatmap;
}

std::vector<float> upscale_field(
    const float* small_field, const std::vector<int>& small_sizes, const std::vector<int>& large_sizes)
{
	VLOG_SCOPE_F(1, "upscale_field");
	CHECK_EQ_F(small_sizes.size(), large_sizes.size());
	int num_dim = small_sizes.size();

	int small_unknowns = 1;
	for (auto small_size : small_sizes) { small_unknowns *= small_size; }

	int large_unknowns = 1;
	for (auto large_size : large_sizes) { large_unknowns *= large_size; }

	std::vector<int> small_strides;
	{
		int small_stride = 1;
		for (int small_size : small_sizes) {
			small_strides.push_back(small_stride);
			small_stride *= small_size;
		}
	}

	std::vector<float> large_field;
	large_field.reserve(large_unknowns);

	for (int large_index = 0; large_index < large_unknowns; ++large_index)
	{
		int large_coordinate[MAX_DIM];
		coordinate_from_index(large_sizes, large_coordinate, large_index);

		float small_pos[MAX_DIM];
		for (int d = 0; d < num_dim; ++d) {
			small_pos[d] = (float)large_coordinate[d] * (small_sizes[d] - 1.0f) / (large_sizes[d] - 1.0f);
		}

		int sample_indices_small[TWO_TO_MAX_DIM];
		float sample_kernel[TWO_TO_MAX_DIM];

		const int num_samples = multilerp(sample_indices_small, sample_kernel, small_sizes, small_strides, small_pos, 0);

		float weight_sum = 0;
		float field_sum = 0;
		for (int i = 0; i < num_samples; ++i) {
			weight_sum += sample_kernel[i];
			field_sum += sample_kernel[i] * small_field[sample_indices_small[i]];
		}

		if (weight_sum == 0) {
			large_field.push_back(0.0f);
		} else {
			large_field.push_back(field_sum / weight_sum);
		}
	}

	return large_field;
}

} // namespace field_interpolation
