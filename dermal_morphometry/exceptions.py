"""Custom exceptions for dermal morphometry pipeline."""

from __future__ import annotations


class DermalMorphometryError(Exception):
    """Base exception for dermal morphometry pipeline failures."""


class NRRDHeaderError(DermalMorphometryError):
    """Raised when a NRRD header is missing, malformed, or unsupported."""


class NRRDDataError(DermalMorphometryError):
    """Raised when NRRD payload bytes cannot be decoded into the declared array."""


class ProcessingError(DermalMorphometryError):
    """Raised when a processing stage cannot produce a valid result."""
