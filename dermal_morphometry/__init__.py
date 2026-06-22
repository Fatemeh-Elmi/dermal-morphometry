"""Dermal fibrous matrix morphometry package for SR-PhC-microCT wound-bed ROIs."""

from __future__ import annotations

from .cli import main
from .constants import (
    HIGH_RES_LIMITATION,
    HIGH_RES_SCIENTIFIC_LABEL,
    LOW_RES_LIMITATION,
    LOW_RES_SCIENTIFIC_LABEL,
    NORM_PERCENTILE_HIGH,
    NORM_PERCENTILE_LOW,
)
from .exceptions import DermalMorphometryError, NRRDDataError, NRRDHeaderError, ProcessingError
from .high_res import HighResConfig, process_high_res
from .low_res import LowResConfig, process_low_res
from .nrrd_io import NRRDVolume, read_nrrd, write_nrrd

__all__ = [
    "DermalMorphometryError",
    "HIGH_RES_LIMITATION",
    "HIGH_RES_SCIENTIFIC_LABEL",
    "HighResConfig",
    "LOW_RES_LIMITATION",
    "LOW_RES_SCIENTIFIC_LABEL",
    "LowResConfig",
    "NORM_PERCENTILE_HIGH",
    "NORM_PERCENTILE_LOW",
    "NRRDDataError",
    "NRRDHeaderError",
    "NRRDVolume",
    "ProcessingError",
    "main",
    "process_high_res",
    "process_low_res",
    "read_nrrd",
    "write_nrrd",
]

__version__ = "1.0.0"
