"""Small smoke test for the dermal_morphometry package."""

from pathlib import Path
import tempfile

import numpy as np

from dermal_morphometry.nrrd_io import read_nrrd, write_nrrd
from dermal_morphometry.low_res import LowResConfig, process_low_res
from dermal_morphometry.high_res import HighResConfig, process_high_res


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rng = np.random.default_rng(10)
        vol = rng.normal(0.2, 0.03, size=(32, 32, 24)).astype(np.float32)
        vol[8:24, 12:18, 6:20] += 0.8
        input_path = root / "synthetic.nrrd"
        write_nrrd(input_path, vol, encoding="gzip")
        reread = read_nrrd(input_path)
        assert reread.data.shape == vol.shape
        process_low_res(input_path, root / "low", LowResConfig(export_mesh=False))
        process_high_res(input_path, root / "high", HighResConfig(downsample=2), sample="synthetic")
        assert (root / "low" / "P86_quantitative_metrics.csv").exists()
        assert (root / "high" / "synthetic_fixed_angle_orientation_class_metrics.csv").exists()


if __name__ == "__main__":
    main()
