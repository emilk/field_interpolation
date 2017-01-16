#pragma once

#include <vector>

#include "sparse_linear.hpp"

/*
This library helps you set up a sparse linear system to estimate a field from a set of constraints.
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

The library is agnostic to dimensionality, but was mainly designed for low-dimensional system.
In particular: 1D, 2D, 3D.

Example use cases:
	* Fit a smooth curve to some data
	* Generate a signed distance field from a set of surface points

The lattice coordinates go from [0, 0, ...] to [width - 1, height - 1, ...] (inclusive).
*/

/// There is no technical limit to this,
/// but note that add_value_constraint adds 2^D equations,
/// and add_gradient_constraint adds 2^(D+1) equations.
/// For this reason you may want to to spread your constraints
/// with nearest-neighbor instead, if your dimensionality is high.
const int MAX_DIM = 4;

struct Strengths
{
	float data_pos      = 1.00f; // How much we trust the point positions
	float data_gradient = 1.00f; // How much we trust the point normals
	float model_0       = 0.00f; // How much we believe the SDF to be zero (regularization).
	float model_1       = 0.00f; // How much we believe the SDF to be uniform.
	float model_2       = 1.00f; // How much we believe the SDF to be smooth.
	float model_3       = 0.00f; // Another order of smoothness.
};

/// Sparse Ax=b where A is described by `triplets` and `rhs` is b.
struct LinearEquation
{
	std::vector<Triplet> triplets;
	std::vector<float>   rhs;
};

struct LinearEquationPair
{
	int   column;
	float value;
};

struct LatticeField
{
	LinearEquation eq;
	int            num_dim;
	int            sizes[MAX_DIM];

	LatticeField(int num_dim_arg, const int sizes_arg[])
	{
		num_dim = num_dim_arg;
		for (int d = 0; d < num_dim; ++d) {
			sizes[d] = sizes_arg[d];
		}
	}
};

/// Helper to add a row to the linear equation.
void add_equation(LinearEquation* eq, float rhs, std::initializer_list<LinearEquationPair> pairs);

/// Add equations describing the model: a smooth field on a lattice.
void add_field_constraints(
	LatticeField*    field,
	const Strengths& strengths);

/// Add a value constraint:  f(pos) = value
/// This is a no-op if pos is close to or outside of the field.
/// Returns false if the position was ignored.
bool add_value_constraint(
	LatticeField* field,
	const float   pos[],
	float         value,
	float         strength);

/// Add a gradient constraint:  ∇ f(pos) = gradient
/// This is a no-op if pos is close to or outside of the field.
/// Returns false if the position was ignored.
bool add_gradient_constraint(
	LatticeField* field,
	const float    pos[],
	const float    gradient[],
	float          strength);

/// Helper function for generating a signed distance field:
LatticeField sdf_from_points(
    int              num_dim,
    const int        sizes[],
    const Strengths& strengths,
    int              num_points,
    const float      positions[], // Interleaved coordinates, e.g. xyxyxy...
    const float*     normals,       // Optional (may be null).
    const float*     point_weights); // Optional (may be null).
