"""Matplotlib visualization utilities, intentionally separated from computations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np


def mip(vol: np.ndarray, axis: int) -> np.ndarray:
    """Maximum-intensity projection along one axis."""
    return np.max(vol, axis=axis)


def add_scale_bar(ax: Axes, voxel_size_um: float, length_um: float, *, label: str | None = None) -> None:
    """Draw a crisp white/black scale bar in image pixel coordinates."""
    if voxel_size_um <= 0:
        return
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    width_px = abs(xlim[1] - xlim[0])
    height_px = abs(ylim[1] - ylim[0])
    bar_px = max(1.0, length_um / voxel_size_um)
    x0 = min(xlim) + 0.06 * width_px
    y0 = max(ylim) - 0.08 * height_px if ylim[0] < ylim[1] else min(ylim) + 0.08 * height_px
    text = label if label is not None else f"{int(length_um)} um"
    ax.plot([x0, x0 + bar_px], [y0, y0], color="white", linewidth=4, solid_capstyle="butt")
    ax.plot([x0, x0 + bar_px], [y0, y0], color="black", linewidth=2, solid_capstyle="butt")
    ax.text(x0, y0 - 0.04 * height_px, text, color="black", fontsize=8, va="top", ha="left",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1})


def colorize_orientation_labels(labels2d: np.ndarray) -> np.ndarray:
    """Convert fixed-angle orientation labels to the manuscript QC color map."""
    colors = np.array(
        [
            [0, 0, 0],          # 0 excluded black/void
            [174, 220, 225],    # 1 low-fiber/background light cyan
            [135, 155, 135],    # 2 oblique/intermediate gray-green
            [255, 205, 0],      # 3 transverse yellow
            [0, 53, 175],       # 4 longitudinal dark blue
        ],
        dtype=np.uint8,
    )
    safe = np.clip(labels2d.astype(np.int16), 0, len(colors) - 1)
    return colors[safe]


def save_low_res_dashboard(
    out_png: Path,
    vol01: np.ndarray,
    signal: np.ndarray,
    mask: np.ndarray,
    skeleton: np.ndarray,
    density: np.ndarray,
    metrics: dict[str, Any],
    *,
    title: str,
    voxel_size_um: float = 3.84,
    scale_bar_um: float = 200.0,
    dpi: int = 300,
) -> None:
    """Create the low-resolution P86 dashboard with standardized scale bars."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13.5, 7.5), dpi=dpi)
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.25])

    panels: Sequence[tuple[str, np.ndarray, str]] = (
        ("Original MIP - axis 0", mip(vol01, axis=0), "gray"),
        ("Original MIP - axis 1", mip(vol01, axis=1), "gray"),
        ("Original MIP - axis 2", mip(vol01, axis=2), "gray"),
        ("Enhanced bright/fibrous signal", mip(signal, axis=0), "gray"),
    )
    for idx, (panel_title, image, cmap) in enumerate(panels):
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(image, cmap=cmap)
        ax.set_title(panel_title, fontsize=9)
        ax.set_xlabel("image x")
        ax.set_ylabel("image y")
        add_scale_bar(ax, voxel_size_um, scale_bar_um)
        ax.set_xticks([])
        ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(mip(vol01, axis=0), cmap="gray")
    overlay = np.ma.masked_where(mip(mask, axis=0) <= 0, mip(mask, axis=0))
    ax.imshow(overlay, cmap="copper", alpha=0.45, vmin=0, vmax=1)
    ax.set_title(f"P86 candidate overlay\ncoverage={metrics['volume_fraction_percent']:.2f}%", fontsize=9)
    add_scale_bar(ax, voxel_size_um, scale_bar_um)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(mip(mask, axis=0), cmap="YlOrRd", vmin=0, vmax=1)
    sk = np.ma.masked_where(mip(skeleton, axis=0) <= 0, mip(skeleton, axis=0))
    ax.imshow(sk, cmap="Reds", alpha=0.85, vmin=0, vmax=1)
    ax.set_title("Slice-wise skeleton QC", fontsize=9)
    add_scale_bar(ax, voxel_size_um, scale_bar_um)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 2])
    im = ax.imshow(mip(density, axis=0), cmap="copper")
    ax.set_title("Local density map", fontsize=9)
    add_scale_bar(ax, voxel_size_um, scale_bar_um)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="local density")

    ax = fig.add_subplot(gs[1, 3])
    ax.axis("off")
    text = (
        "P86 quantitative QC\n"
        f"Mask voxels: {metrics['mask_voxels']}\n"
        f"Volume fraction: {metrics['volume_fraction_percent']:.2f}%\n"
        f"Components: {metrics['component_count']}\n"
        f"Component density / 1e6 voxels: {metrics['component_density_per_1e6_voxels']:.2f}\n"
        f"Largest component: {metrics['largest_component_fraction_percent_of_mask']:.2f}% of mask\n"
        f"Skeleton density: {metrics['skeleton_density_fraction_of_mask']:.4f}\n"
        f"Threshold: {metrics['threshold_value']:.4f}"
    )
    ax.text(0.02, 0.98, text, va="top", ha="left", fontsize=9, family="monospace")
    ax.set_title("Metrics", fontsize=10, fontweight="bold")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def save_low_res_overlay(out_png: Path, vol01: np.ndarray, mask: np.ndarray, *, voxel_size_um: float = 3.84) -> None:
    """Save a standalone maximum-projection P86 overlay."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    ax.imshow(mip(vol01, axis=0), cmap="gray")
    overlay = np.ma.masked_where(mip(mask, axis=0) <= 0, mip(mask, axis=0))
    ax.imshow(overlay, cmap="copper", alpha=0.5, vmin=0, vmax=1)
    add_scale_bar(ax, voxel_size_um, 200.0)
    ax.set_title("P86 fibrous candidate overlay")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def save_orientation_dashboard(
    out_png: Path,
    vol01: np.ndarray,
    labels: np.ndarray,
    metrics: dict[str, Any],
    *,
    title: str,
    voxel_size_um: float = 0.9,
    scale_bar_um: float = 50.0,
    dpi: int = 300,
) -> None:
    """Create high-resolution fixed-angle orientation-class dashboard."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    zdim = labels.shape[0]
    zs = [max(0, zdim // 6), zdim // 2, min(zdim - 1, int(5 * zdim / 6))]
    label_mip = np.max(labels, axis=0)

    fig = plt.figure(figsize=(13.5, 10), dpi=dpi)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 1.20])

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(np.max(vol01, axis=0), cmap="gray")
    ax.set_title("Original MIP", fontsize=9)
    add_scale_bar(ax, voxel_size_um, scale_bar_um)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(colorize_orientation_labels(label_mip))
    ax.set_title("Orientation-class MIP", fontsize=9)
    add_scale_bar(ax, voxel_size_um, scale_bar_um)
    ax.set_xticks([])
    ax.set_yticks([])

    for i, z in enumerate(zs):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(colorize_orientation_labels(labels[z]))
        ax.set_title(f"Slice z={z}", fontsize=9)
        add_scale_bar(ax, voxel_size_um, scale_bar_um)
        ax.set_xticks([])
        ax.set_yticks([])

    ax = fig.add_subplot(gs[0:2, 3])
    ax.axis("off")
    legend_items = [
        ("0 black/void excluded", [0, 0, 0]),
        ("1 low-fiber/background", [174 / 255, 220 / 255, 225 / 255]),
        ("2 oblique/intermediate", [135 / 255, 155 / 255, 135 / 255]),
        ("3 transverse", [1.0, 205 / 255, 0]),
        ("4 longitudinal", [0, 53 / 255, 175 / 255]),
    ]
    y = 0.92
    ax.text(0.02, y, "Legend", fontsize=11, fontweight="bold", transform=ax.transAxes)
    y -= 0.08
    for text, col in legend_items:
        ax.add_patch(plt.Rectangle((0.02, y - 0.02), 0.06, 0.04, color=col, transform=ax.transAxes))
        ax.text(0.11, y, text, fontsize=8.5, va="center", transform=ax.transAxes)
        y -= 0.07

    ax = fig.add_subplot(gs[2, 0:2])
    class_names = ["Excluded", "Low/background", "Oblique", "Transverse", "Longitudinal"]
    vals = [
        metrics["excluded_black_or_void_percent"],
        metrics["low_fiber_background_percent_total"],
        metrics["oblique_intermediate_percent_total"],
        metrics["transverse_percent_total"],
        metrics["longitudinal_percent_total"],
    ]
    ax.bar(class_names, vals)
    ax.set_ylabel("% total voxels")
    ax.set_title("Class fractions")
    ax.tick_params(axis="x", rotation=25)

    ax = fig.add_subplot(gs[2, 2])
    split_names = ["Oblique", "Transverse", "Longitudinal"]
    split_vals = [
        metrics["oblique_fraction_of_fiber_candidate_percent"],
        metrics["transverse_fraction_of_fiber_candidate_percent"],
        metrics["longitudinal_fraction_of_fiber_candidate_percent"],
    ]
    ax.bar(split_names, split_vals)
    ax.set_ylabel("% fiber candidate")
    ax.set_title("Orientation split")
    ax.tick_params(axis="x", rotation=25)

    ax = fig.add_subplot(gs[2, 3])
    ax.axis("off")
    text = (
        f"Excluded black/void: {metrics['excluded_black_or_void_percent']:.2f}%\n"
        f"Valid dermis: {metrics['valid_dermis_percent']:.2f}%\n"
        f"Fiber candidate: {metrics['fiber_candidate_percent_total']:.2f}%\n"
        f"T/L ratio: {metrics['transverse_to_longitudinal_ratio']:.2f}\n"
        f"L/T ratio: {metrics['longitudinal_to_transverse_ratio']:.2f}\n"
        f"Mean coherence: {metrics['mean_orientation_coherence_in_fiber']:.3f}\n"
        f"Alignment imbalance: {metrics['alignment_imbalance_index_abs_L_minus_T_over_fiber']:.3f}"
    )
    ax.text(0.02, 0.98, "Key metrics", fontsize=11, fontweight="bold", va="top")
    ax.text(0.02, 0.82, text, fontsize=9, va="top", family="monospace")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
