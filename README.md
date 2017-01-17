# Linear least-squares field interpolation
A novel(?) method for interpolating sparse and/or noisy data in one or several dimensions.

Data interpolation using sparse linear least squares on a lattice.

Like finite elements. Similar to https://en.wikipedia.org/wiki/Thin_plate_spline.
The constraints form a [boundary value problem](https://en.wikipedia.org/wiki/Boundary_value_problem)

More info coming soon.

# Uses
* Generate iso-fields from point samples, e.g. for mesh reconstruction from point clouds.
* Take sparse samples or noisy samples of a signal and generate a dense Look-Up Table (LUT) with smoothly interpolated values.

# Future improvmenets
* Generalize to non-scalar fields.

# TODO
* Name this method and library
* 1D ImGui with very simple model and graph and low-res
	* Use to verify iso-surface positioning is perfect
* Visualize certainty (abs-gradient should be 1.0 if trustable)
* Persist parameters after startup
* Try dual contouring
