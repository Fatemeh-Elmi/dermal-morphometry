"""Low-resolution P86 fibrous-candidate extraction workflow.

Implements the manuscript low-resolution SR-PhC-microCT analysis track for
wound-bed ROI quantification: intensity normalization, P86 thresholding,
morphological refinement, connectivity metrics, and QC visualizations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import binary_fill_holes, gaussian_filter
from skimage import measure, morphology
from skimage.morphology import ball, remove_small_objects

from .constants import (
    LOW_RES_LIMITATION,
    LOW_RES_SCIENTIFIC_LABEL,
    NORM_PERCENTILE_HIGH,
    NORM_PERCENTILE_LOW,
)
from .exceptions import ProcessingError
from .export import write_json_metadata, write_metrics_csv, write_text_readme, zip_output_directory
from .mesh_io import export_mesh_stl_ply
from .nrrd_io import read_nrrd, write_nrrd
from .preprocessing import robust_normalize01
from .visualization import save_low_res_dashboard, save_low_res_overlay

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LowResConfig:
    """Parameter set for the low-resolution P86 workflow."""

    percentile: int = 86
    sigma: float = 1.0
    min_size: int = 120
    closing_radius: int = 1
    density_sigma: float = 5.0
    voxel_size_um: float = 3.84
    scale_bar_um: float = 200.0
    nrrd_encoding: str = "gzip"
    export_mesh: bool = True


def make_p86_mask(signal: np.ndarray, percentile: int = 86, min_size: int = 120, closing_radius: int = 1) -> tuple[np.ndarray, float]:
    """Create a P86 candidate mask from the enhanced signal.

    The operation order matches the manuscript Methods (Section 2.3):
    percentile threshold, 3D small-object removal, 3D ball closing, and binary
    hole filling.
    """
    threshold_value = float(np.percentile(signal, percentile))
    mask = signal > threshold_value

    if min_size and min_size > 0:
        # scikit-image >= 0.26: max_size removes components with size <= max_size.
        mask = remove_small_objects(mask.astype(bool), max_size=min_size - 1)

    if closing_radius and closing_radius > 0:
        mask = morphology.closing(mask, ball(closing_radius))

    mask = binary_fill_holes(mask)
    return mask.astype(np.uint8), threshold_value


def make_slice_wise_skeleton(mask: np.ndarray) -> np.ndarray:
    """Make a slice-wise 2D skeleton QC volume.

    Slice-wise 2D skeletonization is applied along axis 2 for centerline QC.
    This output is a quality-control map, not a direct fibril-length measurement.
    """
    skel = np.zeros_like(mask, dtype=np.uint8)
    if mask.ndim != 3:
        raise ProcessingError(f"Expected a 3D mask for skeletonization; got ndim={mask.ndim}.")
    for z in range(mask.shape[2]):
        skel[:, :, z] = morphology.skeletonize(mask[:, :, z] > 0).astype(np.uint8)
    return skel


def component_metrics(mask: np.ndarray) -> dict[str, Any]:
    """Return component summaries for the binary P86 mask."""
    labels = measure.label(mask > 0, connectivity=1)
    component_count = int(labels.max())
    counts = np.bincount(labels.ravel())
    if len(counts) > 1:
        component_sizes = counts[1:]
        largest_component_voxels = int(component_sizes.max())
        top5 = sorted((int(x) for x in component_sizes.tolist()), reverse=True)[:5]
    else:
        largest_component_voxels = 0
        top5 = []

    mask_voxels = int(mask.sum())
    total_voxels = int(mask.size)
    largest_component_fraction = float(largest_component_voxels / max(mask_voxels, 1) * 100.0)
    component_density = float(component_count / max(total_voxels, 1) * 1_000_000.0)

    return {
        "component_count": component_count,
        "component_density_per_1e6_voxels": component_density,
        "largest_component_voxels": largest_component_voxels,
        "largest_component_fraction_percent_of_mask": largest_component_fraction,
        "top5_component_voxels": top5,
    }


def compute_low_res_metrics(mask: np.ndarray, skeleton: np.ndarray, density: np.ndarray, signal: np.ndarray, threshold_value: float, percentile: int) -> dict[str, Any]:
    """Compute P86 low-resolution metrics used by the manuscript tables."""
    mask_voxels = int(mask.sum())
    total_voxels = int(mask.size)
    volume_fraction_percent = float(mask_voxels / max(total_voxels, 1) * 100.0)
    skel_voxels = int(skeleton.sum())

    metrics: dict[str, Any] = {
        "percentile": int(percentile),
        "threshold_value": float(threshold_value),
        "mask_voxels": mask_voxels,
        "total_voxels": total_voxels,
        "volume_fraction_percent": volume_fraction_percent,
        "skeleton_voxels_slice_wise": skel_voxels,
        "skeleton_density_fraction_of_mask": float(skel_voxels / max(mask_voxels, 1)),
        "mean_density_in_mask": float(density[mask > 0].mean()) if mask_voxels > 0 else 0.0,
        "mean_enhanced_signal_in_mask": float(signal[mask > 0].mean()) if mask_voxels > 0 else 0.0,
        "mean_enhanced_signal_background": float(signal[mask == 0].mean()) if mask_voxels < total_voxels else 0.0,
    }
    metrics.update(component_metrics(mask))
    return metrics


def process_low_res(input_path: Path, outdir: Path, config: LowResConfig) -> dict[str, Any]:
    """Run the complete low-resolution P86 workflow."""
    outdir.mkdir(parents=True, exist_ok=True)
    if config.percentile != 86:
        LOGGER.warning("The study pipeline uses P86 thresholding; requested percentile=%s.", config.percentile)

    volume = read_nrrd(input_path)
    vol = volume.data
    vol01, norm_low, norm_high = robust_normalize01(vol, NORM_PERCENTILE_LOW, NORM_PERCENTILE_HIGH)
    signal = gaussian_filter(vol01, sigma=config.sigma).astype(np.float32)
    mask, threshold_value = make_p86_mask(signal, percentile=config.percentile, min_size=config.min_size, closing_radius=config.closing_radius)
    skeleton = make_slice_wise_skeleton(mask)
    density = gaussian_filter(mask.astype(np.float32), sigma=config.density_sigma).astype(np.float32)
    metrics = compute_low_res_metrics(mask, skeleton, density, signal, threshold_value, config.percentile)

    write_nrrd(outdir / "P86_enhanced_fibrous_signal_bright_density.nrrd", signal.astype(np.float32), volume.header, encoding=config.nrrd_encoding)
    write_nrrd(outdir / "P86_candidate_mask_uint8.nrrd", mask.astype(np.uint8), volume.header, encoding=config.nrrd_encoding)
    write_nrrd(outdir / "P86_skeleton_slice_wise_uint8.nrrd", skeleton.astype(np.uint8), volume.header, encoding=config.nrrd_encoding)
    write_nrrd(outdir / "P86_local_density_map_float32.nrrd", density.astype(np.float32), volume.header, encoding=config.nrrd_encoding)

    write_metrics_csv(outdir / "P86_quantitative_metrics.csv", metrics)
    save_low_res_dashboard(
        outdir / "P86_scientific_qc_dashboard.png",
        vol01,
        signal,
        mask,
        skeleton,
        density,
        metrics,
        title="P86 scientific QC - collagen-rich / fibrous candidate",
        voxel_size_um=config.voxel_size_um,
        scale_bar_um=config.scale_bar_um,
    )
    save_low_res_overlay(outdir / "P86_MIP_overlay.png", vol01, mask, voxel_size_um=config.voxel_size_um)

    mesh_info: dict[str, Any] | dict[str, str] | None = None
    if config.export_mesh:
        mesh_info = export_mesh_stl_ply(
            mask,
            volume.header,
            outdir / "P86_surface_mesh.stl",
            outdir / "P86_surface_mesh.ply",
            solid_name="P86_candidate_mask",
        )

    metadata = {
        "input_file": str(input_path),
        "mode": "low-res",
        "volume_shape_voxels": list(vol.shape),
        "dtype": str(vol.dtype),
        "normalization_percentile_clip": [NORM_PERCENTILE_LOW, NORM_PERCENTILE_HIGH],
        "normalization_clip_values": [norm_low, norm_high],
        "scientific_label": LOW_RES_SCIENTIFIC_LABEL,
        "important_limitation": LOW_RES_LIMITATION,
        "config": asdict(config),
        "mesh": mesh_info,
        "metrics": metrics,
    }
    write_json_metadata(outdir / "P86_scientific_metadata.json", metadata)

    readme = f"""P86-only low-resolution method summary
======================================

Input: {input_path.name}

Scientific label: collagen-rich / fibrous connective tissue candidate.
This is a reproducible phase-contrast fibrous candidate mask, not a chemically
specific collagen segmentation without external validation.

Processing parity:
- Normalization: 1st-99th percentile clipping to [0, 1]
- Enhanced signal: Gaussian smoothing sigma={config.sigma}
- Threshold: P{config.percentile} of enhanced signal
- Remove small objects: min_size={config.min_size} voxels
- 3D morphological closing: ball radius={config.closing_radius} voxel
- Binary hole filling: enabled
- Local density: Gaussian smoothing of the binary mask, sigma={config.density_sigma}
- Skeleton QC: slice-wise 2D skeletonization along axis 2

Primary metrics:
- P86 coverage: {metrics['volume_fraction_percent']:.4f}%
- Connected components: {metrics['component_count']}
- Component density per 1e6 voxels: {metrics['component_density_per_1e6_voxels']:.4f}
- Largest component fraction: {metrics['largest_component_fraction_percent_of_mask']:.4f}% of mask
- Skeleton density: {metrics['skeleton_density_fraction_of_mask']:.6f}
"""
    write_text_readme(outdir / "README_P86_method.txt", readme)
    zip_output_directory(outdir)
    LOGGER.info("Low-resolution workflow complete: %s", outdir)
    return metadata
