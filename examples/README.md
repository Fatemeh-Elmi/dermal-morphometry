# Examples

Runnable demonstrations using a small synthetic NRRD volume. These examples do not require study ROI data.

## Quick start

From the package root, after installation:

```bash
python examples/run_synthetic_demo.py
```

Outputs are written to `examples/output/low_res/` and `examples/output/high_res/`.

## What the demo does

1. Builds a 32 × 32 × 24 synthetic float32 volume with a bright fibrous-like ROI.
2. Saves it as `examples/data/synthetic_roi.nrrd`.
3. Runs the low-resolution P86 workflow (mesh export disabled for speed).
4. Runs the high-resolution orientation workflow with `--sample synthetic`.

## Run on your own ROI

Replace the synthetic input with a cropped NRRD exported from 3D Slicer:

```bash
dermal-morphometry --mode low-res --input path/to/ROI_low.nrrd --outdir examples/output/my_low -v
dermal-morphometry --mode high-res --input path/to/ROI_high.nrrd --outdir examples/output/my_high --sample chitosan_G90_ROI1 -v
```

Low- and high-resolution ROIs normally come from separate acquisitions (Section 2.3).

## Expected outputs

**Low-resolution:** `P86_quantitative_metrics.csv`, `P86_scientific_qc_dashboard.png`, NRRD masks, metadata JSON, ZIP archive.

**High-resolution:** `synthetic_fixed_angle_orientation_class_metrics.csv`, orientation-class dashboard, label volumes, metadata JSON, ZIP archive.
