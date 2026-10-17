
"""Generate a synthetic ROI and run both analysis workflows."""

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


def main():
    data_dir = Path(__file__).parent / "data"
    output_dir = Path(__file__).parent / "output"

    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(10)

    vol = rng.normal(
        0.2, 0.03, size=(32, 32, 24)
    ).astype(np.float32)

    vol[8:24, 12:18, 6:20] += 0.8

    input_path = data_dir / "synthetic_roi.nrrd"
    write_nrrd(input_path, vol, encoding="gzip")

    low_meta = process_low_res(
        input_path,
        output_dir / "low_res",
        LowResConfig(export_mesh=False),
    )

    print(
        f"Low-res volume fraction: "
        f"{low_meta['metrics']['volume_fraction_percent']:.2f}%"
    )

    high_meta = process_high_res(
        input_path,
        output_dir / "high_res",
        HighResConfig(),
        sample="synthetic",
    )

    fiber_pct = high_meta["metrics"]["fiber_candidate_percent_total"]

    print(f"High-res P60/P55 fiber candidate fraction: {fiber_pct:.2f}%")
    print(f"Done. See {output_dir}")


if __name__ == "__main__":
    main()
