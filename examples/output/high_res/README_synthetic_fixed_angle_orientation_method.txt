High-resolution fixed-angle orientation method summary
=====================================================

Input: synthetic_roi.nrrd
Sample: synthetic

Processing parity:
- Preserve reconstruction axes; no rotation or reorientation
- Downsample by integer stride: 2
- Normalize: 1st-99th percentile clipping to [0, 1]
- Exclude very black/void voxels: black percentile=7.0
- Candidate mask: intensity percentile=86.0, texture percentile=70.0, texture weight=0.35, min object size=8
- 2D slice-wise structure tensor: sigma_gradient=1.0, sigma_tensor=2.0
- Longitudinal: 0 +/- 30.0 degrees or 180 +/- 30.0 degrees
- Transverse: 90 +/- 30.0 degrees
- Remaining fiber candidates: oblique/intermediate

Primary metrics:
- Fiber candidate total: 6.1523%
- Transverse total: 0.1302%
- Longitudinal total: 5.9570%
- L/T ratio: 45.7500
- Mean coherence in fiber: 0.483201
