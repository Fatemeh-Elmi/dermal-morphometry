"""Unified command-line interface for dermal morphometry workflows."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from .exceptions import DermalMorphometryError
from .high_res import HighResConfig, process_high_res
from .low_res import LowResConfig, process_low_res


def configure_logging(verbosity: int) -> None:
    """Configure root logging from -v flags."""
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dermal-morphometry",
        description="Unified SR-PhC-microCT dermal fibrous candidate and orientation analysis pipeline.",
    )
    parser.add_argument("--input", required=True, help="Input cropped 3D NRRD volume.")
    parser.add_argument("--outdir", required=True, help="Output directory.")
    parser.add_argument("--mode", choices=("low-res", "high-res", "both"), default="both", help="Workflow to run.")
    parser.add_argument("--sample", default="sample", help="Sample label used for high-resolution output names.")
    parser.add_argument("--nrrd-encoding", choices=("gzip", "raw"), default="gzip", help="Encoding for generated NRRD outputs.")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase logging verbosity; repeat for debug logging.")

    low = parser.add_argument_group("low-resolution P86 options")
    low.add_argument("--low-percentile", type=int, default=86, help="Low-res final threshold percentile; manuscript default P86.")
    low.add_argument("--low-sigma", type=float, default=1.0, help="Low-res Gaussian smoothing sigma.")
    low.add_argument("--low-min-size", type=int, default=120, help="Low-res 3D small-object removal size in voxels.")
    low.add_argument("--low-closing-radius", type=int, default=1, help="Low-res 3D morphological closing radius in voxels.")
    low.add_argument("--low-density-sigma", type=float, default=5.0, help="Low-res Gaussian sigma for local density map.")
    low.add_argument("--low-voxel-size-um", type=float, default=3.84, help="Low-res voxel size in um for scale bars.")
    low.add_argument("--no-mesh", action="store_true", help="Skip low-res STL/PLY mesh export.")

    high = parser.add_argument_group("high-resolution orientation options")
    high.add_argument("--downsample", type=int, default=6, help="High-res integer downsample factor (manuscript default 6).")
    high.add_argument("--high-voxel-size-um", type=float, default=0.9, help="High-res voxel size in um (manuscript acquisition 0.9 um).")
    high.add_argument("--black-percentile", type=float, default=7.0, help="High-res black/void exclusion percentile.")
    high.add_argument("--p-intensity", type=float, default=86.0, help="High-res intensity percentile for candidates.")
    high.add_argument("--texture-percentile", type=float, default=70.0, help="High-res texture percentile gate.")
    high.add_argument("--texture-weight", type=float, default=0.35, help="Weight of texture score in candidate extraction.")
    high.add_argument("--high-min-size", type=int, default=8, help="High-res minimum object size in voxels.")
    high.add_argument("--angle-tolerance", type=float, default=30.0, help="Fixed-angle tolerance in degrees.")
    high.add_argument("--sigma-gradient", type=float, default=1.0, help="Structure tensor pre-gradient Gaussian sigma.")
    high.add_argument("--sigma-tensor", type=float, default=2.0, help="Structure tensor smoothing sigma.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        if args.mode in ("low-res", "both"):
            low_outdir = outdir if args.mode == "low-res" else outdir / "low_res"
            low_config = LowResConfig(
                percentile=args.low_percentile,
                sigma=args.low_sigma,
                min_size=args.low_min_size,
                closing_radius=args.low_closing_radius,
                density_sigma=args.low_density_sigma,
                voxel_size_um=args.low_voxel_size_um,
                nrrd_encoding=args.nrrd_encoding,
                export_mesh=not args.no_mesh,
            )
            process_low_res(input_path, low_outdir, low_config)

        if args.mode in ("high-res", "both"):
            high_outdir = outdir if args.mode == "high-res" else outdir / "high_res"
            high_config = HighResConfig(
                downsample=args.downsample,
                voxel_size_um=args.high_voxel_size_um,
                black_percentile=args.black_percentile,
                p_intensity=args.p_intensity,
                texture_percentile=args.texture_percentile,
                texture_weight=args.texture_weight,
                min_object_size_voxels=args.high_min_size,
                angle_tolerance=args.angle_tolerance,
                sigma_gradient=args.sigma_gradient,
                sigma_tensor=args.sigma_tensor,
                nrrd_encoding=args.nrrd_encoding,
            )
            process_high_res(input_path, high_outdir, high_config, sample=args.sample)
    except DermalMorphometryError as exc:
        logging.getLogger(__name__).error("Pipeline failed: %s", exc)
        return 2

    logging.getLogger(__name__).info("All requested workflows completed in %s", outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
