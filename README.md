# Linear least-squares field interpolation
A novel(?) method for interpolating sparse and/or noisy data in one or several dimensions.

Data interpolation using sparse linear least squares on a lattice.

Like finite elements. Similar to https://en.wikipedia.org/wiki/Thin_plate_spline.
The constraints form a [boundary value problem](https://en.wikipedia.org/wiki/Boundary_value_problem)

Keywords:
interpolation, sparse linear least squares, lattice, boundary problem, finite element method.

More info coming soon.

# Uses
* Generate iso-fields from point samples, e.g. for mesh reconstruction from point clouds.
* Take sparse samples or noisy samples of a signal and generate a dense Look-Up Table (LUT) with smoothly interpolated values.

# Future improvmenets
* Generalize to non-scalar fields (e.g. RGB).


# TODO
* Name this method and library. Finite Element Interpolation ? emilerp?

## Algo
* Add nearest-neighbor versions, mostly to make them easy to read and comprehend.
* Try dual contouring
* Figure out the correct approach to gradients. Sqrt? Single-equation? Summed the sqrt weight? Etc.

## Speeding up
* Coarse first, scale up, re-run on elements with large errors, keep smooth fixed (sparse)
* Coarse first, scale up, improve with gauss-siedel or Conjugate Gradient

## Gui
* Use 1D to verify iso-surface positioning is perfect
* Split back-projected error into model and data constraints.
* Add several saved configs for distance field tab
