# Finite element field interpolation
A method for interpolating sparse and/or noisy data in one or several dimensions.
Can be used to generate a signed distance field from a point cloud.

# Uses
* Generate iso-fields from point samples, e.g. for mesh reconstruction from point clouds.
* Take sparse samples and/or noisy samples of a signal and generate a dense Look-Up Table (LUT) with smoothly interpolated values.

# TODO
* Name this method and library. Finite Element Interpolation ? emilerp?

## Algo
* Add nearest-neighbor versions, mostly to make them easy to read and comprehend.

## Speeding up
* Coarse first, scale up, re-run on elements with large errors, keep smooth fixed (sparse)
* Coarse first, scale up, improve with gauss-siedel or Conjugate Gradient

## Gui
* Use 1D to verify iso-surface positioning is perfect
* Split back-projected error into model and data constraints.
* Add several saved configs for distance field tab
