"""Preprocessing utilities shared by low- and high-resolution workflows."""

from __future__ import annotations

import numpy as np


def robust_normalize01(vol: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> tuple[np.ndarray, float, float]:
    """Clip at the requested percentiles and rescale robustly to [0, 1]."""
    arr = np.asarray(vol, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32), 0.0, 0.0
    lo, hi = np.percentile(finite, [p_low, p_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr, dtype=np.float32), float(lo), float(hi)
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return out, float(lo), float(hi)


def downsample_volume(vol: np.ndarray, factor: int = 1) -> np.ndarray:
    """Downsample a volume by integer stride while preserving original axis order."""
    if factor <= 1:
        return vol
    return vol[::factor, ::factor, ::factor]
