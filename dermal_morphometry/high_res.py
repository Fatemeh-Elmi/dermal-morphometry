
"""High-resolution dermal fibrous-candidate and orientation analysis.

Manuscript settings:
- Original ROI resolution (DS1)
- P1/P99 intensity normalization
- Valid tissue: normalized intensity > ROI P7
- Local 3x3x3 intensity standard deviation
- Segmentation: valid AND (intensity >= P60 OR texture >= P55)
- No small-object removal or morphological post-processing
- 2D structure tensor on YZ slices (NRRD axes X,Y,Z)
- Orientation classes relative to image axes, not anatomical axes
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import re

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
from .preprocessing import robust_normalize01
from .visualization import save_orientation_dashboard


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class HighResConfig:
    downsample: int = 1
    black_percentile: float = 7.0
    p_intensity: float = 60.0
    texture_percentile: float = 55.0
    angle_tolerance: float = 30.0
    sigma_gradient: float = 1.0
    sigma_tensor: float = 2.0
    voxel_size_um: float | None = None
    scale_bar_um: float = 50.0
    nrrd_encoding: str = "gzip"


def verify_image_axes(header: dict) -> None:
    """Check axis-aligned X,Y,Z geometry when available."""
    directions = header.get("space directions", "")

    if not directions:
        return

    triples = re.findall(r"\(([^)]*)\)", str(directions))

    if len(triples) != 3:
        raise ValueError("Unsupported NRRD space directions")

    vectors = np.array(
        [
            [float(value) for value in triple.split(",")]
            for triple in triples
        ],
        dtype=float,
    )

    if vectors.shape != (3, 3):
        raise ValueError("Invalid NRRD spatial geometry")

    if not np.all(np.isfinite(vectors)):
        raise ValueError("Non-finite spatial coordinates")

    if np.any(np.diag(vectors) == 0):
        raise ValueError("Expected X,Y,Z axis order")

    off_diagonal = vectors - np.diag(np.diag(vectors))

    if not np.allclose(off_diagonal, 0, atol=1e-6):
        raise ValueError(
            "Non-axis-aligned NRRD requires geometry verification"
        )


def local_standard_deviation(
    vol01: np.ndarray,
) -> np.ndarray:
    """Local intensity SD using a reflected 3x3x3 neighborhood."""

    image = vol01.astype(np.float64)

    local_mean = ndi.uniform_filter(
        image,
        size=3,
        mode="reflect",
    )

    local_second_moment = ndi.uniform_filter(
        image * image,
        size=3,
        mode="reflect",
    )

    local_variance = np.maximum(
        local_second_moment - local_mean ** 2,
        0.0,
    )

    return np.sqrt(local_variance)


def fibrous_candidate_mask(
    vol01: np.ndarray,
    valid_mask: np.ndarray,
    intensity_percentile: float = 60.0,
    texture_percentile: float = 55.0,
) -> np.ndarray:
    """Apply the P60/P55 logical-OR segmentation rule."""

    if not np.any(valid_mask):
        raise ValueError("Empty valid tissue mask")

    texture = local_standard_deviation(vol01)

    intensity_threshold = np.percentile(
        vol01[valid_mask],
        intensity_percentile,
    )

    texture_threshold = np.percentile(
        texture[valid_mask],
        texture_percentile,
    )

    fiber_mask = valid_mask & (
        (vol01 >= intensity_threshold)
        | (texture >= texture_threshold)
    )

    return fiber_mask


def structure_tensor_orientation_2d(
    image: np.ndarray,
    sigma_gradient: float = 1.0,
    sigma_tensor: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate structural angle and coherence on a YZ slice."""

    smooth = ndi.gaussian_filter(
        image.astype(np.float32),
        sigma=sigma_gradient,
    )

    gradient_y, gradient_z = np.gradient(smooth)

    jzz = ndi.gaussian_filter(
        gradient_z * gradient_z,
        sigma=sigma_tensor,
    )

    jyy = ndi.gaussian_filter(
        gradient_y * gradient_y,
        sigma=sigma_tensor,
    )

    jyz = ndi.gaussian_filter(
        gradient_y * gradient_z,
        sigma=sigma_tensor,
    )

    theta = (
        np.degrees(
            0.5 * np.arctan2(
                2.0 * jyz,
                jzz - jyy,
            ) + np.pi / 2.0
        ) % 180.0
    )

    coherence = np.sqrt(
        (jzz - jyy) ** 2 + 4.0 * jyz ** 2
    ) / (jzz + jyy + 1e-8)

    coherence = np.clip(coherence, 0.0, 1.0)

    return (
        theta.astype(np.float32),
        coherence.astype(np.float32),
    )


def classify_fixed_angle(
    angle: np.ndarray,
    fiber_mask: np.ndarray,
    valid_mask: np.ndarray,
    tolerance_deg: float = 30.0,
) -> np.ndarray:
    """Classify image-coordinate orientation.

    0 = excluded
    1 = valid noncandidate
    2 = oblique
    3 = Z-axis aligned (transverse)
    4 = Y-axis aligned (longitudinal)
    """

    labels = np.zeros(angle.shape, dtype=np.uint8)

    labels[valid_mask] = 1
    labels[fiber_mask] = 2

    z_aligned = fiber_mask & (
        (angle <= tolerance_deg)
        | (angle >= 180.0 - tolerance_deg)
    )

    y_aligned = fiber_mask & (
        np.abs(angle - 90.0) <= tolerance_deg
    )

    labels[z_aligned] = 3
    labels[y_aligned] = 4

    return labels


def estimate_orientation_volume(
    vol01: np.ndarray,
    fiber_mask: np.ndarray,
    valid_mask: np.ndarray,
    tolerance_deg: float = 30.0,
    sigma_gradient: float = 1.0,
    sigma_tensor: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process 2D YZ slices along the X axis."""

    angle = np.zeros(vol01.shape, dtype=np.float32)
    coherence = np.zeros(vol01.shape, dtype=np.float32)
    labels = np.zeros(vol01.shape, dtype=np.uint8)

    for x in range(vol01.shape[0]):

        a, c = structure_tensor_orientation_2d(
            vol01[x],
            sigma_gradient=sigma_gradient,
            sigma_tensor=sigma_tensor,
        )

        angle[x] = a
        coherence[x] = c

        labels[x] = classify_fixed_angle(
            a,
            fiber_mask[x],
            valid_mask[x],
            tolerance_deg=tolerance_deg,
        )

    return angle, coherence, labels


def compute_orientation_metrics(
    labels: np.ndarray,
    coherence: np.ndarray,
    sample: str,
    input_file: str,
    tolerance_deg: float,
    original_shape: tuple[int, int, int],
) -> dict:
    """Calculate ROI-level quantitative measurements."""

    total = labels.size

    excluded_n = int(np.count_nonzero(labels == 0))
    background_n = int(np.count_nonzero(labels == 1))
    oblique_n = int(np.count_nonzero(labels == 2))
    transverse_n = int(np.count_nonzero(labels == 3))
    longitudinal_n = int(np.count_nonzero(labels == 4))

    fiber_n = (
        oblique_n + transverse_n + longitudinal_n
    )

    def percent(n):
        return float(100.0 * n / total)

    return {
        "sample": sample,
        "input_file": input_file,
        "method": "P60/P55 OR; DS1; YZ structure tensor",
        "downsample_factor": 1,
        "fixed_angle_tolerance_deg": tolerance_deg,
        "original_shape": str(original_shape),
        "analyzed_shape": str(labels.shape),

        "excluded_black_or_void_percent":
            percent(excluded_n),

        "valid_dermis_percent":
            percent(total - excluded_n),

        "low_fiber_background_percent_total":
            percent(background_n),

        "fiber_candidate_percent_total":
            percent(fiber_n),

        "oblique_intermediate_percent_total":
            percent(oblique_n),

        "transverse_percent_total":
            percent(transverse_n),

        "longitudinal_percent_total":
            percent(longitudinal_n),

        "oblique_fraction_of_fiber_candidate_percent":
            100.0 * oblique_n / fiber_n
            if fiber_n else None,

        "transverse_fraction_of_fiber_candidate_percent":
            100.0 * transverse_n / fiber_n
            if fiber_n else None,

        "longitudinal_fraction_of_fiber_candidate_percent":
            100.0 * longitudinal_n / fiber_n
            if fiber_n else None,

        "transverse_to_longitudinal_ratio":
            transverse_n / longitudinal_n
            if longitudinal_n else None,

        "longitudinal_to_transverse_ratio":
            longitudinal_n / transverse_n
            if transverse_n else None,

        "mean_orientation_coherence_in_fiber":
            float(np.mean(coherence[labels >= 2]))
            if fiber_n else None,

        "alignment_imbalance_index_abs_L_minus_T_over_fiber":
            abs(longitudinal_n - transverse_n) / fiber_n
            if fiber_n else None,
    }


def process_high_res(
    input_path: Path,
    outdir: Path,
    config: HighResConfig,
    sample: str = "sample",
) -> dict:
    """Run the manuscript high-resolution analysis."""

    input_path = Path(input_path)
    outdir = Path(outdir)

    if config.downsample != 1:
        raise ValueError(
            "High-resolution manuscript analysis requires DS1"
        )

    if (
        config.p_intensity != 60.0
        or config.texture_percentile != 55.0
    ):
        raise ValueError(
            "The manuscript profile requires P60/P55"
        )

    outdir.mkdir(parents=True, exist_ok=True)

    volume = read_nrrd(input_path)

    original = volume.data
    header = volume.header

    if original.ndim != 3:
        raise ValueError("Input must be a 3D NRRD volume")

    verify_image_axes(header)

    vol01, norm_low, norm_high = robust_normalize01(
        original,
        NORM_PERCENTILE_LOW,
        NORM_PERCENTILE_HIGH,
    )

    black_threshold = float(
        np.percentile(vol01, config.black_percentile)
    )

    valid_mask = vol01 > black_threshold

    fiber_mask = fibrous_candidate_mask(
        vol01,
        valid_mask,
        intensity_percentile=config.p_intensity,
        texture_percentile=config.texture_percentile,
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

    outputs = {
        f"{stem}_fiber_candidate_ds1_uint8.nrrd":
            fiber_mask.astype(np.uint8),

        f"{stem}_valid_tissue_ds1_uint8.nrrd":
            valid_mask.astype(np.uint8),

        f"{stem}_fixed_angle_orientation_class_labels_ds1_uint8.nrrd":
            labels,

        f"{stem}_structure_tensor_angle_deg_ds1_float32.nrrd":
            angle,

        f"{stem}_structure_tensor_coherence_ds1_float32.nrrd":
            coherence,
    }

    for filename, array in outputs.items():
        write_nrrd(
            outdir / filename,
            array,
            header,
            encoding=config.nrrd_encoding,
        )

    metrics = compute_orientation_metrics(
        labels,
        coherence,
        sample=sample,
        input_file=input_path.name,
        tolerance_deg=config.angle_tolerance,
        original_shape=tuple(original.shape),
    )

    metrics_path = (
        outdir
        / f"{stem}_fixed_angle_orientation_class_metrics.csv"
    )

    write_metrics_csv(metrics_path, metrics)

    # A scale bar is drawn only with verified voxel size.
    if (
        config.voxel_size_um is not None
        and config.voxel_size_um > 0
    ):
        save_orientation_dashboard(
            outdir
            / f"{stem}_fixed_angle_orientation_class_fastQC_dashboard.png",
            vol01=vol01,
            labels=labels,
            metrics=metrics,
            title=f"{sample} | YZ image-coordinate orientation",
            voxel_size_um=config.voxel_size_um,
            scale_bar_um=config.scale_bar_um,
        )

    metadata = {
        "input_file": str(input_path),
        "mode": "high-res",
        "sample": sample,
        "original_shape": tuple(int(x) for x in original.shape),
        "analyzed_shape": tuple(int(x) for x in labels.shape),
        "normalization_percentile_clip": [
            NORM_PERCENTILE_LOW,
            NORM_PERCENTILE_HIGH,
        ],
        "normalization_clip_values": [
            float(norm_low),
            float(norm_high),
        ],
        "black_threshold": black_threshold,
        "scientific_label": HIGH_RES_SCIENTIFIC_LABEL,
        "important_limitation": HIGH_RES_LIMITATION,
        "config": asdict(config),
        "metrics": metrics,
    }

    write_json_metadata(
        outdir
        / f"{stem}_fixed_angle_orientation_metadata.json",
        metadata,
    )

    readme = f"""High-resolution orientation method summary
================================================

Input: {input_path.name}
Sample: {sample}

Processing settings:
- Original resolution: DS1
- No rotation or reorientation
- Intensity normalization: P1/P99
- Valid tissue: normalized intensity > ROI P7
- Intensity threshold: P60
- Texture threshold: P55
- Texture: 3x3x3 local intensity standard deviation
- Segmentation: valid AND (intensity >= P60 OR texture >= P55)
- No weighted score
- No connected-component or morphological filtering
- Tensor planes: YZ slices along X
- Sigma gradient: {config.sigma_gradient}
- Sigma tensor: {config.sigma_tensor}
- Angle tolerance: +/-{config.angle_tolerance} degrees
- Longitudinal: Y image axis
- Transverse: Z image axis
- Denominator: all original ROI voxels
- Orientation classes are not verified anatomical directions

Primary metrics:
- Fiber candidate: {metrics['fiber_candidate_percent_total']:.4f}%
- Oblique: {metrics['oblique_intermediate_percent_total']:.4f}%
- Transverse: {metrics['transverse_percent_total']:.4f}%
- Longitudinal: {metrics['longitudinal_percent_total']:.4f}%
- L/T ratio: {metrics['longitudinal_to_transverse_ratio']}
- Mean coherence: {metrics['mean_orientation_coherence_in_fiber']}
"""

    write_text_readme(
        outdir
        / f"README_{stem}_fixed_angle_orientation_method.txt",
        readme,
    )

    zip_output_directory(outdir)

    LOGGER.info(
        "High-resolution P60/P55 DS1 completed: %s",
        outdir,
    )

    return metadata

