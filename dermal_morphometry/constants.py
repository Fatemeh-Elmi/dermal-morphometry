"""Shared constants and manuscript-parity defaults."""

from __future__ import annotations

# Robust intensity normalization applied in both workflows.
NORM_PERCENTILE_LOW: float = 1.0
NORM_PERCENTILE_HIGH: float = 99.0

# Terminology used consistently in outputs and metadata.
LOW_RES_SCIENTIFIC_LABEL: str = "collagen-rich / fibrous connective tissue candidate"
HIGH_RES_SCIENTIFIC_LABEL: str = "fibrous/collagen-rich candidate; image-axis orientation descriptors"

LOW_RES_LIMITATION: str = (
    "This is not chemically definitive collagen without validation by staining, "
    "FTIR, SHG, or another reference method."
)
HIGH_RES_LIMITATION: str = (
    "Longitudinal and transverse labels are fixed image-axis classes, "
    "not verified in-vivo anatomical collagen directions."
)
