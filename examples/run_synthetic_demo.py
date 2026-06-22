"""Generate a synthetic ROI and run both analysis workflows.

Run from the package root:

    python examples/run_synthetic_demo.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dermal_morphometry.high_res import HighResConfig, process_high_res
from dermal_morphometry.low_res import LowResConfig, process_low_res
from dermal_morphometry.nrrd_io import write_nrrd


def make_synthetic_volume(shape: tuple[int, int, int] = (32, 32, 24), seed: int = 10) -> np.ndarray:
    """Create a small volume with background noise and a bright fibrous-like block."""
    rng = np.random.default_rng(seed)
    vol = rng.normal(0.2, 0.03, size=shape).astype(np.float32)
    z0, z1 = 6, min(shape[2], 20)
    y0, y1 = 12, min(shape[1], 18)
    x0, x1 = 8, min(shape[0], 24)
    vol[x0:x1, y0:y1, z0:z1] += 0.8
    return vol


def main() -> None:
    data_dir = Path(__file__).resolve().parent / "data"
    output_dir = Path(__file__).resolve().parent / "output"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = data_dir / "synthetic_roi.nrrd"
    vol = make_synthetic_volume()
    write_nrrd(input_path, vol, encoding="gzip")
    print(f"Wrote synthetic input: {input_path}")

    low_meta = process_low_res(
        input_path,
        output_dir / "low_res",
        LowResConfig(export_mesh=False),
    )
    print(f"Low-res volume fraction: {low_meta['metrics']['volume_fraction_percent']:.2f}%")

    high_meta = process_high_res(
        input_path,
        output_dir / "high_res",
        HighResConfig(downsample=2),
        sample="synthetic",
    )
    fiber_pct = high_meta["metrics"]["fiber_candidate_percent_total"]
    print(f"High-res fiber candidate fraction: {fiber_pct:.2f}%")
    print(f"Done. See {output_dir}")


if __name__ == "__main__":
    main()
