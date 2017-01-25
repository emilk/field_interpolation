#pragma once

#include <iosfwd>
#include <vector>

#include "sparse_linear.hpp"

/*
This library helps you set up a sparse linear system to estimate a field from a set of constraints.
The field will be represented by a square lattice (a grid).
There are two types of constraint:
	* Model constraint. These are equations describing the smoothness of the field.
	* Data constraints. This is specific things we know about the field. Can be:
		* The value of the field at a given position
		* The gradient of the field at a given position.

By using a sparse least squares linear solver you will get a field which
minimizes the errors of these constraints.

Each constraint can have a weight. In particular,
your data points can have high weights if they are trustworthy or low otherwise.
Your model constraint weights may be high if you trust that the field is smooth.
You can thus tweak the relative weights for:
	* A lot of noisy data (high smoothness weight, low data weight)
	* Sparse and accurate data (low smoothness weight, hight data weight)
	* Sharpness/smoothness of model (soft hills or sharp mountains?)

The library is agnostic to dimensionality, but was mainly designed for low-dimensional systems.
In particular: 1D, 2D, 3D.

Example use cases:
	* Fit a smooth curve to some data
	* Generate a signed distance field (sdf) from a set of surface points

The lattice coordinates go from [0, 0, ...] to [width - 1, height - 1, ...] (inclusive).
*/

/// There is no technical limit to this,
/// but note that add_value_constraint adds 2^D equations,
/// and add_gradient_constraint adds 2^(D+1) equations.
/// For this reason you may want to to spread your constraints
/// with nearest-neighbor instead, if your dimensionality is high.
const int MAX_DIM = 4;

extern bool g_nn_gradient;

/// A note about picking good parameters:
/// If your model is continuous but with abrupt changes, use a high model_1 and low everything else.
/// If your model is smooth, use a high model_2 and low everything else.
/// If your data is trustworthy, you should lower the model weights (e.g. 1/10th of the data weights).
/// If your data is noisy, you should use higher model weights.
/// If your data is lopsided (a lot of points in one area, fewer in another) you should lower model_1.
/// Note that if you increase the resolution of your lattice, you should modify the model weights.
/// In particular:
///     model_0 = constant_0 * resolution
///     model_1 = constant_1
///     model_2 = constant_2 / resolution
///     model_3 = constant_3 / resolution^2
/// Where resolution is e.g. the width of your lattice.
/// Higher orders of smoothness increases the computational cost!
struct Weights
{
	float data_pos      = 1.00f; ///< How much we trust the point value/position
	float data_gradient = 1.00f; ///< How much we trust the point gradient/normal
	// https://en.wikipedia.org/wiki/Smoothness#Order_of_continuity
	float model_0       = 0.00f; ///< How much we believe the field to be zero (regularization). If this is large everything will be zero.
	float model_1       = 0.10f; ///< How much we believe the field to be uniform. If this is large you will take the average of the data.
	float model_2       = 0.50f; ///< How much we believe the field to be smooth. If this is large you will be fitting a line to the data.
	float model_3       = 0.00f; ///< If this is large, you will be fitting a quadratic curve to you data.
	float model_4       = 0.00f; ///< If this is large, you will be fitting a cubic curve to you data.
};

/// Sparse Ax=b where A is described by `triplets` and `rhs` is b.
struct LinearEquation
{
	std::vector<Triplet> triplets;
	std::vector<float>   rhs;
};

std::ostream& operator<<(std::ostream& os, const LinearEquation& eq);

struct LinearEquationPair
{
	int   column;
	float value;
};

struct LatticeField
{
	LinearEquation   eq;      ///< Accumulated equations.
	std::vector<int> sizes;   ///< sizes[d] == size of dimension `d`
	std::vector<int> strides; ///< stride[d] == distance between adjacent values along dimension `d`

	LatticeField() = default;
	explicit LatticeField(const std::vector<int>& sizes_arg) : sizes(sizes_arg)
	{
		int stride = 1;
		for (int size : sizes) {
			strides.push_back(stride);
			stride *= size;
		}
	}
};

struct Weight { float value; };
struct Rhs    { float value; };

/// Helper to add a row to the linear equation.
void add_equation(
	LinearEquation* eq, Weight weight, Rhs rhs, std::initializer_list<LinearEquationPair> pairs);

/// Add equations describing the model: a smooth field on a lattice.
void add_field_constraints(
	LatticeField*  field,
	const Weights& weights);

/// Add a value constraint:  f(pos) = value
/// This is a no-op if pos is close to or outside of the field.
/// Returns false if the position was ignored.
bool add_value_constraint(
	LatticeField* field,
	const float   pos[],
	float         value,
	float         weight);

/// Add a gradient constraint:  ∇ f(pos) = gradient
/// This is a no-op if pos is close to or outside of the field.
/// Returns false if the position was ignored.
bool add_gradient_constraint(
	LatticeField* field,
	const float    pos[],
	const float    gradient[],
	float          weight);

/// Helper function for generating a signed distance field:
/// The resulting distances may be scaled arbitrarily, and only accurate near field=0.
/// Still, it will be useful for finding the field=0 surface using e.g. marching cubes.
LatticeField sdf_from_points(
	const std::vector<int>& sizes,          // Lattice size: one for each dimension
	const Weights&          weights,
	const int               num_points,
	const float             positions[],    // Interleaved coordinates, e.g. xyxyxy...
	const float*            normals,        // Optional (may be null).
	const float*            point_weights); // Optional (may be null).

/// Calculate (Ax - b)^2 and distribute onto the solution space for a heatmap of blame.
std::vector<float> generate_error_map(
	const std::vector<Triplet>& triplets,
	const std::vector<float>&   solution,
	const std::vector<float>&   rhs);
