"""Lightweight independent NRRD reader/writer for 3D Slicer-style volumes.

Supports attached-data NRRD files with raw, gzip, gz, and txt/text/ascii
encodings. Implemented internally so the analysis package has no external NRRD
library dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .exceptions import NRRDDataError, NRRDHeaderError

LOGGER = logging.getLogger(__name__)


NRRD_TO_NUMPY: dict[str, str] = {
    "uchar": "u1",
    "unsigned char": "u1",
    "uint8": "u1",
    "uint8_t": "u1",
    "signed char": "i1",
    "int8": "i1",
    "int8_t": "i1",
    "short": "i2",
    "short int": "i2",
    "signed short": "i2",
    "signed short int": "i2",
    "int16": "i2",
    "int16_t": "i2",
    "ushort": "u2",
    "unsigned short": "u2",
    "unsigned short int": "u2",
    "uint16": "u2",
    "uint16_t": "u2",
    "int": "i4",
    "signed int": "i4",
    "int32": "i4",
    "int32_t": "i4",
    "uint": "u4",
    "unsigned int": "u4",
    "uint32": "u4",
    "uint32_t": "u4",
    "longlong": "i8",
    "long long": "i8",
    "long long int": "i8",
    "int64": "i8",
    "int64_t": "i8",
    "ulonglong": "u8",
    "unsigned long long": "u8",
    "uint64": "u8",
    "uint64_t": "u8",
    "float": "f4",
    "single": "f4",
    "double": "f8",
}

NUMPY_TO_NRRD: dict[str, str] = {
    "uint8": "uchar",
    "int8": "signed char",
    "uint16": "ushort",
    "int16": "short",
    "uint32": "uint",
    "int32": "int",
    "uint64": "uint64",
    "int64": "int64",
    "float32": "float",
    "float64": "double",
}


@dataclass(frozen=True)
class NRRDVolume:
    """Container for an NRRD array and normalized lowercase header fields."""

    data: np.ndarray
    header: dict[str, str]


def _split_header_payload(raw: bytes) -> tuple[str, bytes]:
    """Split attached NRRD bytes into ASCII header text and payload bytes."""
    separators = [(b"\n\n", 2), (b"\r\n\r\n", 4)]
    for sep, sep_len in separators:
        idx = raw.find(sep)
        if idx >= 0:
            try:
                return raw[:idx].decode("ascii", errors="strict"), raw[idx + sep_len :]
            except UnicodeDecodeError as exc:
                raise NRRDHeaderError("NRRD header is not valid ASCII.") from exc
    raise NRRDHeaderError("Could not find NRRD header/data separator.")


def _parse_header(header_text: str) -> dict[str, str]:
    """Parse an NRRD header into lowercase fields while preserving values."""
    lines = header_text.splitlines()
    if not lines or not lines[0].startswith("NRRD"):
        raise NRRDHeaderError("File does not start with an NRRD magic line.")

    fields: dict[str, str] = {}
    for raw_line in lines[1:]:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise NRRDHeaderError(f"Malformed NRRD header line: {raw_line!r}")
        key, value = line.split(":", 1)
        fields[key.strip().lower()] = value.strip()

    required = ("type", "dimension", "sizes")
    missing = [key for key in required if key not in fields]
    if missing:
        raise NRRDHeaderError(f"NRRD header missing required field(s): {', '.join(missing)}")

    try:
        dimension = int(fields["dimension"])
    except ValueError as exc:
        raise NRRDHeaderError("NRRD dimension field is not an integer.") from exc
    if dimension != 3:
        raise NRRDHeaderError(f"Only 3D NRRD volumes are supported; found dimension={dimension}.")

    sizes = fields["sizes"].split()
    if len(sizes) != dimension:
        raise NRRDHeaderError("NRRD sizes field length does not match dimension.")
    try:
        tuple(int(x) for x in sizes)
    except ValueError as exc:
        raise NRRDHeaderError("NRRD sizes field contains non-integer values.") from exc

    return fields


def _dtype_from_header(fields: dict[str, str]) -> np.dtype[Any]:
    typ = fields.get("type", "").strip().lower()
    base = NRRD_TO_NUMPY.get(typ)
    if base is None:
        raise NRRDHeaderError(f"Unsupported NRRD type: {typ!r}")

    dtype = np.dtype(base)
    endian = fields.get("endian", "little").strip().lower()
    if dtype.itemsize > 1:
        if endian in ("little", "littleendian"):
            dtype = dtype.newbyteorder("<")
        elif endian in ("big", "bigendian"):
            dtype = dtype.newbyteorder(">")
        else:
            raise NRRDHeaderError(f"Unsupported NRRD endian value: {endian!r}")
    return dtype


def _decode_payload(payload: bytes, fields: dict[str, str]) -> bytes:
    encoding = fields.get("encoding", "raw").strip().lower()
    if encoding in ("raw", ""):
        return payload
    if encoding in ("gzip", "gz"):
        try:
            return gzip.decompress(payload)
        except OSError as exc:
            raise NRRDDataError("Could not decompress gzip-encoded NRRD payload.") from exc
    if encoding in ("txt", "text", "ascii"):
        return payload
    raise NRRDHeaderError(f"Unsupported NRRD encoding: {encoding!r}")


def read_nrrd(path: str | Path) -> NRRDVolume:
    """Read a 3D attached-data NRRD file.

    Data are reshaped with Fortran order to remain compatible with 3D Slicer
    NRRD exports used in the study workflow.
    """
    nrrd_path = Path(path)
    if not nrrd_path.exists():
        raise FileNotFoundError(nrrd_path)

    LOGGER.debug("Reading NRRD file: %s", nrrd_path)
    header_text, payload = _split_header_payload(nrrd_path.read_bytes())
    fields = _parse_header(header_text)
    payload = _decode_payload(payload, fields)

    dtype = _dtype_from_header(fields)
    sizes = tuple(int(x) for x in fields["sizes"].split())
    encoding = fields.get("encoding", "raw").strip().lower()

    if encoding in ("txt", "text", "ascii"):
        try:
            arr = np.loadtxt(payload.decode("ascii").splitlines(), dtype=dtype)
        except Exception as exc:  # noqa: BLE001 - normalize external parsing failures
            raise NRRDDataError("Could not parse ASCII NRRD payload.") from exc
        arr = np.asarray(arr, dtype=dtype)
    else:
        expected_bytes = int(np.prod(sizes)) * dtype.itemsize
        if len(payload) < expected_bytes:
            raise NRRDDataError(
                f"NRRD payload too small: expected {expected_bytes} bytes, got {len(payload)} bytes."
            )
        arr = np.frombuffer(payload[:expected_bytes], dtype=dtype)

    expected_voxels = int(np.prod(sizes))
    if arr.size != expected_voxels:
        raise NRRDDataError(f"NRRD payload voxel count mismatch: expected {expected_voxels}, got {arr.size}.")

    data = np.asarray(arr.reshape(sizes, order="F"))
    LOGGER.info("Loaded NRRD %s shape=%s dtype=%s", nrrd_path.name, data.shape, data.dtype)
    return NRRDVolume(data=data, header=fields)


def _nrrd_type_for_array(arr: np.ndarray) -> str:
    key = str(np.dtype(arr.dtype).newbyteorder("=")).replace("<", "").replace(">", "")
    if key not in NUMPY_TO_NRRD:
        raise NRRDHeaderError(f"Cannot write unsupported array dtype as NRRD: {arr.dtype}")
    return NUMPY_TO_NRRD[key]


def _copy_spatial_fields(header: dict[str, str] | None) -> dict[str, str]:
    source = {} if header is None else {str(k).lower(): str(v) for k, v in header.items()}
    copied: dict[str, str] = {}
    for key in ("space", "space directions", "space origin", "measurement frame"):
        if key in source:
            copied[key] = source[key]
    return copied


def write_nrrd(
    path: str | Path,
    arr: np.ndarray,
    header: dict[str, str] | None = None,
    *,
    encoding: str = "gzip",
    compresslevel: int = 1,
) -> None:
    """Write a 3D attached-data NRRD file.

    The payload uses Fortran order, matching 3D Slicer-compatible NRRD exports.
    Use ``encoding='raw'`` for uncompressed output or ``encoding='gzip'`` for
    standard Slicer-readable compressed output.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise NRRDHeaderError(f"Only 3D arrays can be written; got ndim={arr.ndim}.")

    encoding_norm = encoding.lower().strip()
    if encoding_norm not in ("raw", "gzip", "gz"):
        raise NRRDHeaderError(f"Unsupported writer encoding: {encoding!r}")
    nrrd_type = _nrrd_type_for_array(arr)
    spatial = _copy_spatial_fields(header)

    header_lines = [
        "NRRD0004",
        f"type: {nrrd_type}",
        "dimension: 3",
        f"sizes: {' '.join(str(int(s)) for s in arr.shape)}",
        "kinds: domain domain domain",
    ]
    if arr.dtype.itemsize > 1:
        header_lines.append("endian: little")
    if "space" in spatial:
        header_lines.append(f"space: {spatial['space']}")
    if "space directions" in spatial:
        header_lines.append(f"space directions: {spatial['space directions']}")
    if "space origin" in spatial:
        header_lines.append(f"space origin: {spatial['space origin']}")
    if "measurement frame" in spatial:
        header_lines.append(f"measurement frame: {spatial['measurement frame']}")

    header_lines.append("encoding: gzip" if encoding_norm in ("gzip", "gz") else "encoding: raw")
    header_bytes = ("\n".join(header_lines) + "\n\n").encode("ascii")
    payload = np.ascontiguousarray(arr).tobytes(order="F")
    if encoding_norm in ("gzip", "gz"):
        payload = gzip.compress(payload, compresslevel=compresslevel)
    out_path.write_bytes(header_bytes + payload)
    LOGGER.info("Wrote NRRD %s shape=%s dtype=%s", out_path.name, arr.shape, arr.dtype)
