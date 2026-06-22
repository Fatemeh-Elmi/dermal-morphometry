P86-only low-resolution method summary
======================================

Input: synthetic_roi.nrrd

Scientific label: collagen-rich / fibrous connective tissue candidate.
This is a reproducible phase-contrast fibrous candidate mask, not a chemically
specific collagen segmentation without external validation.

Processing parity:
- Normalization: 1st-99th percentile clipping to [0, 1]
- Enhanced signal: Gaussian smoothing sigma=1.0
- Threshold: P86 of enhanced signal
- Remove small objects: min_size=120 voxels
- 3D morphological closing: ball radius=1 voxel
- Binary hole filling: enabled
- Local density: Gaussian smoothing of the binary mask, sigma=5.0
- Skeleton QC: slice-wise 2D skeletonization along axis 2

Primary metrics:
- P86 coverage: 13.4766%
- Connected components: 1
- Component density per 1e6 voxels: 40.6901
- Largest component fraction: 100.0000% of mask
- Skeleton density: 0.065821
