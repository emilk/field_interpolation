#include "field_interpolation.hpp"

#include <cmath>
#include <ostream>

#include <loguru.hpp>

const int TWO_TO_MAX_DIM = (1 << 4);

bool g_nn_gradient = false;

std::ostream& operator<<(std::ostream& os, const LinearEquation& eq)
{
	const size_t num_rows = eq.rhs.size();
	std::vector<std::vector<Triplet>> row_triplets(num_rows);

	for (const auto& triplet : eq.triplets) {
		row_triplets[triplet.row].push_back(triplet);
	}

	for (size_t row = 0; row < num_rows; ++row) {
		os << eq.rhs[row] << " = ";
		for (size_t triplet_idx = 0; triplet_idx < row_triplets[row].size(); ++triplet_idx) {
			const auto& triplet = row_triplets[row][triplet_idx];
			os << triplet.value << " * x" << triplet.col;
			if (triplet_idx + 1 < row_triplets[row].size()) {
				os  << "  +  ";
			}
		}
		os << "\n";
	}
	return os;
}

void add_equation(
	LinearEquation* eq, Weight weight, Rhs rhs, std::initializer_list<LinearEquationPair> pairs)
{
	if (weight.value == 0) { return; }

	// bool all_zero = rhs == 0;
	bool all_zero = true;
	int row = eq->rhs.size();
	for (const auto& pair : pairs) {
		if (pair.value != 0) {
			eq->triplets.emplace_back(row, pair.column, pair.value * weight.value);
			all_zero = false;
		}
	}
	if (!all_zero) {
		eq->rhs.emplace_back(rhs.value * weight.value);
	}
}

/// Computed coefficients for multi-dimensional linear interpolation of 2^D neighbors.
/// Returns the number of indices to sample from.
/// The indices are put in out_indices, the kernel (inteprolation weights) in out_kernel.
int multilerp(
	int                 out_indices[],
	float               out_kernel[],
	const LatticeField& field,
	const float         in_pos[],
	int                 extra_bound)
{
	int num_dim = field.sizes.size();
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
			index  += field.strides[d] * dim_coord;
			weight *= (set ? t[d] : 1.0f - t[d]);
			inside &= (0 <= dim_coord && dim_coord + extra_bound < field.sizes[d]);
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

	int inteprolation_indices[TWO_TO_MAX_DIM];
	float interpolation_kernel[TWO_TO_MAX_DIM];
	int num_samples = multilerp(inteprolation_indices, interpolation_kernel, *field, pos, 0);
	if (num_samples == 0) { return false; }

	int row = field->eq.rhs.size();
	float weight_sum = 0;
	for (int i = 0; i < num_samples; ++i) {
		float sample_weight = interpolation_kernel[i] * constraint_weight;
		field->eq.triplets.emplace_back(row, inteprolation_indices[i], sample_weight);
		weight_sum += sample_weight;
	}
	field->eq.rhs.emplace_back(weight_sum * value);

	return true;
}

bool add_gradient_constraint(
	LatticeField* field,
	const float   pos[],
	const float   gradient[],
	float         constraint_weight)
{
	if (constraint_weight == 0) { return false; }

	if (g_nn_gradient) {
		int num_dim = field->sizes.size();

		int index = 0;
		for (int d = 0; d < num_dim; ++d) {
			int pos_d = std::floor(pos[d]);
			bool in_lattice = 0 <= pos_d && pos_d + 1 < field->sizes[d];
			if (!in_lattice) { return false; }
			index += pos_d * field->strides[d];
		}
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
	} else {
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

		int num_dim = field->sizes.size();

		float adjusted_pos[MAX_DIM];
		for (int d = 0; d < num_dim; ++d) {
			adjusted_pos[d] = pos[d] - 0.5f;
		}

		int inteprolation_indices[TWO_TO_MAX_DIM];
		float interpolation_kernel[TWO_TO_MAX_DIM];
		int num_samples = multilerp(inteprolation_indices, interpolation_kernel, *field, adjusted_pos, 1);
		if (num_samples == 0) { return false; }

		for (int d = 0; d < num_dim; ++d) {
			int row = field->eq.rhs.size();
			float weight_sum = 0;
			for (int i = 0; i < num_samples; ++i) {
				// d f(x, y) / dx = gradient[0]
				// d f(x, y) / dy = gradient[1]
				// ...
				const float sample_weight = interpolation_kernel[i] * constraint_weight;
				field->eq.triplets.emplace_back(row, inteprolation_indices[i] + 0,                 -sample_weight);
				field->eq.triplets.emplace_back(row, inteprolation_indices[i] + field->strides[d], +sample_weight);
				weight_sum += sample_weight;
			}
			field->eq.rhs.emplace_back(weight_sum * gradient[d]);
		}

		return true;
	}
}

/// Add smoothness constraints between the unknowns:  index - stride, index, index + stride, ...
void add_model_constraint(
	LatticeField*  field,
	const Weights& weights,
	int            index,    // index of this value
	int            dim_cord, // coordinate on this dimension
	int            d)        // dimension
{
	const int size   = field->sizes[d];
	const int stride = field->strides[d];

	// TODO: dynamically compute Pascals triangle for this?

	if (0 <= dim_cord && dim_cord < size) {
		// f(x) = 0
		// Tikhonov diagonal regularization
		add_equation(&field->eq, Weight{weights.model_0}, Rhs{0.0f}, {
			{index, 1.0f},
		});
	}

	if (0 <= dim_cord && dim_cord + 1 < size) {
		// f′(x) = 0   ⇔   f(x) = f(x + 1)
		add_equation(&field->eq, Weight{weights.model_1}, Rhs{0.0f}, {
			{index + 0 * stride, -1.0f},
			{index + 1 * stride, +1.0f},
		});
	}

	if (0 <= dim_cord && dim_cord + 2 < size) {
		// f″(x) = 0   ⇔   f′(x - ½) = f′(x + ½)
		add_equation(&field->eq, Weight{weights.model_2}, Rhs{0.0f}, {
			{index + 0 * stride, +1.0f},
			{index + 1 * stride, -2.0f},
			{index + 2 * stride, +1.0f},
		});
	}

	if (0 <= dim_cord && dim_cord + 3 < size) {
		// f‴(x) = 0   ⇔   f″(x - ½) = f″(x + ½)
		add_equation(&field->eq, Weight{weights.model_3}, Rhs{0.0f}, {
			{index + 0 * stride, +1.0f},
			{index + 1 * stride, -3.0f},
			{index + 2 * stride, +3.0f},
			{index + 3 * stride, -1.0f},
		});
	}

	if (0 <= dim_cord && dim_cord + 4 < size) {
		// f⁗(x) = 0   ⇔   f‴(x - ½) = f‴(x + ½)
		add_equation(&field->eq, Weight{weights.model_4}, Rhs{0.0f}, {
			{index + 0 * stride, +1.0f},
			{index + 1 * stride, -4.0f},
			{index + 2 * stride, +6.0f},
			{index + 3 * stride, -4.0f},
			{index + 4 * stride, +1.0f},
		});
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
		int stride = 1;
		int coordinate = index;
		for (int d = 0; d < field->sizes.size(); ++d) {
			int dim_cord = coordinate % field->sizes[d];
			add_model_constraint(field, weights, index, dim_cord, d);
			coordinate /= field->sizes[d];
			stride *= field->sizes[d];
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
	LOG_SCOPE_F(INFO, "sdf_from_points");
	CHECK_NOTNULL_F(positions);

	int num_dim = sizes.size();
	LatticeField field{sizes};

	add_field_constraints(&field, weights);

	for (int i = 0; i < num_points; ++i) {
		float weight = point_weights ? point_weights[i] : 1.0f;
		const float* pos = positions + i * num_dim;
		add_value_constraint(&field, pos, 0.0f, weight * weights.data_pos);
		if (normals) {
			add_gradient_constraint(&field, pos, normals + i * num_dim, weight * weights.data_gradient);
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
