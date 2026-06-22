"""High-resolution fixed-angle orientation-class workflow.

Implements the manuscript high-resolution SR-PhC-microCT analysis track:
fibrous-candidate extraction, slice-wise 2D structure-tensor orientation
estimation, fixed-angle class assignment, and ROI-level orientation metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from .constants import (
    HIGH_RES_LIMITATION,
    HIGH_RES_SCIENTIFIC_LABEL,
    NORM_PERCENTILE_HIGH,
    NORM_PERCENTILE_LOW,
)
from .export import (
    safe_output_stem,
    write_json_metadata,
    write_metrics_csv,
    write_text_readme,
    zip_output_directory,
)
from .nrrd_io import read_nrrd, write_nrrd
from .preprocessing import downsample_volume, robust_normalize01
from .visualization import save_orientation_dashboard

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class HighResConfig:
    """Parameter set for the high-resolution orientation workflow."""

    downsample: int = 6
    voxel_size_um: float = 0.9
    black_percentile: float = 7.0
    p_intensity: float = 86.0
    texture_percentile: float = 70.0
    texture_weight: float = 0.35
    min_object_size_voxels: int = 8
    angle_tolerance: float = 30.0
    sigma_gradient: float = 1.0
    sigma_tensor: float = 2.0
    scale_bar_um: float = 50.0
    nrrd_encoding: str = "gzip"


def compute_texture_score(vol01: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Compute the smoothed gradient-magnitude texture proxy used for candidate scoring."""
    sm = ndi.gaussian_filter(vol01, sigma=sigma)
    gz, gy, gx = np.gradient(sm)
    grad_mag = np.sqrt(gx * gx + gy * gy + gz * gz)
    texture, _lo, _hi = robust_normalize01(grad_mag, NORM_PERCENTILE_LOW, NORM_PERCENTILE_HIGH)
    return texture


def fibrous_candidate_mask(
    vol01: np.ndarray,
    valid_mask: np.ndarray,
    intensity_percentile: float = 86.0,
    texture_percentile: float = 70.0,
    texture_weight: float = 0.35,
    min_object_size_voxels: int = 8,
) -> np.ndarray:
    """Extract high-resolution fibrous candidates using intensity and texture gates."""
    texture = compute_texture_score(vol01, sigma=1.0)
    score = (1.0 - texture_weight) * vol01 + texture_weight * texture
    valid_values = score[valid_mask]
    if valid_values.size == 0:
        return np.zeros_like(valid_mask, dtype=bool)

    thr_intensity = float(np.percentile(vol01[valid_mask], intensity_percentile))
    thr_score = float(np.percentile(valid_values, intensity_percentile))
    thr_texture = float(np.percentile(texture[valid_mask], texture_percentile))

    mask = valid_mask & (vol01 >= thr_intensity) & ((score >= thr_score) | (texture >= thr_texture))
    if min_object_size_voxels > 1:
        lab, n = ndi.label(mask)
        if n > 0:
            sizes = np.bincount(lab.ravel())
            keep = sizes >= min_object_size_voxels
            keep[0] = False
            mask = keep[lab]
    return mask.astype(bool)


def structure_tensor_orientation_2d(img2d: np.ndarray, sigma_gradient: float = 1.0, sigma_tensor: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Estimate 2D structure orientation angle and coherence in one XY slice."""
    img = img2d.astype(np.float32, copy=False)
    sm = ndi.gaussian_filter(img, sigma=sigma_gradient)
    gy, gx = np.gradient(sm)

    jxx = ndi.gaussian_filter(gx * gx, sigma=sigma_tensor)
    jyy = ndi.gaussian_filter(gy * gy, sigma=sigma_tensor)
    jxy = ndi.gaussian_filter(gx * gy, sigma=sigma_tensor)

    theta_grad = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    theta_struct = theta_grad + np.pi / 2.0
    angle_deg = (np.degrees(theta_struct) % 180.0).astype(np.float32)

    trace = jxx + jyy
    delta = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2)
    coherence = np.divide(delta, trace + 1e-8)
    coherence = np.clip(coherence, 0.0, 1.0).astype(np.float32)
    return angle_deg, coherence


def classify_fixed_angle(angle_deg: np.ndarray, fiber_mask: np.ndarray, valid_mask: np.ndarray, tolerance_deg: float = 30.0) -> np.ndarray:
    """Assign fixed-angle orientation classes relative to the image x-axis."""
    labels = np.zeros(angle_deg.shape, dtype=np.uint8)
    labels[valid_mask] = 1
    labels[fiber_mask] = 2
    longitudinal = fiber_mask & ((angle_deg <= tolerance_deg) | (angle_deg >= 180.0 - tolerance_deg))
    transverse = fiber_mask & (np.abs(angle_deg - 90.0) <= tolerance_deg)
    labels[transverse] = 3
    labels[longitudinal] = 4
    return labels


def estimate_orientation_volume(
    vol01: np.ndarray,
    fiber_mask: np.ndarray,
    valid_mask: np.ndarray,
    tolerance_deg: float = 30.0,
    sigma_gradient: float = 1.0,
    sigma_tensor: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate slice-wise 2D orientation in a Z,Y,X volume."""
    zdim = vol01.shape[0]
    angle = np.zeros_like(vol01, dtype=np.float32)
    coherence = np.zeros_like(vol01, dtype=np.float32)
    labels = np.zeros_like(vol01, dtype=np.uint8)
    for z in range(zdim):
        a, c = structure_tensor_orientation_2d(vol01[z], sigma_gradient=sigma_gradient, sigma_tensor=sigma_tensor)
        angle[z] = a
        coherence[z] = c
        labels[z] = classify_fixed_angle(a, fiber_mask[z], valid_mask[z], tolerance_deg=tolerance_deg)
    return angle, coherence, labels


def compute_orientation_metrics(
    labels: np.ndarray,
    coherence: np.ndarray,
    sample: str,
    input_file: str,
    tolerance_deg: float,
    downsample_factor: int,
    voxel_size_um_assumed: float,
    original_shape: tuple[int, int, int],
) -> dict[str, Any]:
    """Compute ROI-level orientation class metrics."""
    total = int(labels.size)
    excluded = labels == 0
    valid = labels != 0
    low = labels == 1
    oblique = labels == 2
    transverse = labels == 3
    longitudinal = labels == 4
    fiber = oblique | transverse | longitudinal

    def pct(mask: np.ndarray, denom: int = total) -> float:
        return float(mask.sum() / max(denom, 1) * 100.0)

    fiber_n = int(fiber.sum())
    transverse_n = int(transverse.sum())
    longitudinal_n = int(longitudinal.sum())
    metrics: dict[str, Any] = {
        "sample": sample,
        "input_file": input_file,
        "method": "structure-tensor fixed-angle; no median split",
        "fixed_angle_tolerance_deg": float(tolerance_deg),
        "downsample_factor": int(downsample_factor),
        "voxel_size_um_assumed": float(voxel_size_um_assumed),
        "original_shape": str(tuple(original_shape)),
        "analyzed_shape": str(tuple(labels.shape)),
        "excluded_black_or_void_percent": pct(excluded),
        "valid_dermis_percent": pct(valid),
        "low_fiber_background_percent_total": pct(low),
        "fiber_candidate_percent_total": pct(fiber),
        "oblique_intermediate_percent_total": pct(oblique),
        "transverse_percent_total": pct(transverse),
        "longitudinal_percent_total": pct(longitudinal),
        "oblique_fraction_of_fiber_candidate_percent": pct(oblique, fiber_n),
        "transverse_fraction_of_fiber_candidate_percent": pct(transverse, fiber_n),
        "longitudinal_fraction_of_fiber_candidate_percent": pct(longitudinal, fiber_n),
        "transverse_to_longitudinal_ratio": float(transverse_n / max(longitudinal_n, 1)),
        "longitudinal_to_transverse_ratio": float(longitudinal_n / max(transverse_n, 1)),
        "mean_orientation_coherence_in_fiber": float(np.nanmean(coherence[fiber])) if fiber_n else float("nan"),
        "alignment_imbalance_index_abs_L_minus_T_over_fiber": float(abs(longitudinal_n - transverse_n) / max(fiber_n, 1)),
    }
    return metrics


def process_high_res(input_path: Path, outdir: Path, config: HighResConfig, sample: str = "sample") -> dict[str, Any]:
    """Run the complete high-resolution fixed-angle workflow."""
    outdir.mkdir(parents=True, exist_ok=True)
    volume = read_nrrd(input_path)
    original_shape = tuple(int(x) for x in volume.data.shape)

    vol_ds = downsample_volume(volume.data, config.downsample)
    vol01, norm_low, norm_high = robust_normalize01(vol_ds, NORM_PERCENTILE_LOW, NORM_PERCENTILE_HIGH)
    finite = vol01[np.isfinite(vol01)]
    black_thr = float(np.percentile(finite, config.black_percentile)) if finite.size else 0.0
    valid_mask = vol01 > black_thr
    fiber_mask = fibrous_candidate_mask(
        vol01,
        valid_mask,
        intensity_percentile=config.p_intensity,
        texture_percentile=config.texture_percentile,
        texture_weight=config.texture_weight,
        min_object_size_voxels=config.min_object_size_voxels,
    )
    angle, coherence, labels = estimate_orientation_volume(
        vol01,
        fiber_mask,
        valid_mask,
        tolerance_deg=config.angle_tolerance,
        sigma_gradient=config.sigma_gradient,
        sigma_tensor=config.sigma_tensor,
    )

    stem = safe_output_stem(sample)
    analyzed_voxel = config.voxel_size_um * max(config.downsample, 1)
    write_nrrd(outdir / f"{stem}_fiber_candidate_ds{config.downsample}_uint8.nrrd", fiber_mask.astype(np.uint8), volume.header, encoding=config.nrrd_encoding)
    write_nrrd(outdir / f"{stem}_valid_dermis_minus_black_ds{config.downsample}_uint8.nrrd", valid_mask.astype(np.uint8), volume.header, encoding=config.nrrd_encoding)
    write_nrrd(outdir / f"{stem}_fixed_angle_orientation_class_labels_ds{config.downsample}_uint8.nrrd", labels.astype(np.uint8), volume.header, encoding=config.nrrd_encoding)
    write_nrrd(outdir / f"{stem}_structure_tensor_angle_deg_ds{config.downsample}_float32.nrrd", angle.astype(np.float32), volume.header, encoding=config.nrrd_encoding)
    write_nrrd(outdir / f"{stem}_structure_tensor_coherence_ds{config.downsample}_float32.nrrd", coherence.astype(np.float32), volume.header, encoding=config.nrrd_encoding)

    metrics = compute_orientation_metrics(
        labels=labels,
        coherence=coherence,
        sample=sample,
        input_file=input_path.name,
        tolerance_deg=config.angle_tolerance,
        downsample_factor=config.downsample,
        voxel_size_um_assumed=analyzed_voxel,
        original_shape=original_shape,
    )
    write_metrics_csv(outdir / f"{stem}_fixed_angle_orientation_class_metrics.csv", metrics)
    save_orientation_dashboard(
        outdir / f"{stem}_fixed_angle_orientation_class_fastQC_dashboard.png",
        vol01=vol01,
        labels=labels,
        metrics=metrics,
        title=f"{sample} fixed-angle orientation-class map",
        voxel_size_um=analyzed_voxel,
        scale_bar_um=config.scale_bar_um,
    )
    metadata = {
        "input_file": str(input_path),
        "mode": "high-res",
        "sample": sample,
        "original_shape": original_shape,
        "analyzed_shape": tuple(int(x) for x in labels.shape),
        "normalization_percentile_clip": [NORM_PERCENTILE_LOW, NORM_PERCENTILE_HIGH],
        "normalization_clip_values": [norm_low, norm_high],
        "black_threshold": black_thr,
        "scientific_label": HIGH_RES_SCIENTIFIC_LABEL,
        "important_limitation": HIGH_RES_LIMITATION,
        "config": asdict(config),
        "metrics": metrics,
    }
    write_json_metadata(outdir / f"{stem}_fixed_angle_orientation_metadata.json", metadata)
    readme = f"""High-resolution fixed-angle orientation method summary
=====================================================

Input: {input_path.name}
Sample: {sample}

Processing parity:
- Preserve reconstruction axes; no rotation or reorientation
- Downsample by integer stride: {config.downsample}
- Normalize: 1st-99th percentile clipping to [0, 1]
- Exclude very black/void voxels: black percentile={config.black_percentile}
- Candidate mask: intensity percentile={config.p_intensity}, texture percentile={config.texture_percentile}, texture weight={config.texture_weight}, min object size={config.min_object_size_voxels}
- 2D slice-wise structure tensor: sigma_gradient={config.sigma_gradient}, sigma_tensor={config.sigma_tensor}
- Longitudinal: 0 +/- {config.angle_tolerance} degrees or 180 +/- {config.angle_tolerance} degrees
- Transverse: 90 +/- {config.angle_tolerance} degrees
- Remaining fiber candidates: oblique/intermediate

Primary metrics:
- Fiber candidate total: {metrics['fiber_candidate_percent_total']:.4f}%
- Transverse total: {metrics['transverse_percent_total']:.4f}%
- Longitudinal total: {metrics['longitudinal_percent_total']:.4f}%
- L/T ratio: {metrics['longitudinal_to_transverse_ratio']:.4f}
- Mean coherence in fiber: {metrics['mean_orientation_coherence_in_fiber']:.6f}
"""
    write_text_readme(outdir / f"README_{stem}_fixed_angle_orientation_method.txt", readme)
    zip_output_directory(outdir)
    LOGGER.info("High-resolution workflow complete: %s", outdir)
    return metadata
