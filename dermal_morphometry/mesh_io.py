"""Optional surface-mesh export from binary segmentation masks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from skimage import measure

LOGGER = logging.getLogger(__name__)


def spacing_from_nrrd_header(header: dict[str, str]) -> tuple[float, float, float]:
    """Extract best-effort voxel spacing from NRRD spatial header fields."""
    space_directions = header.get("space directions", "")
    if not space_directions:
        return (1.0, 1.0, 1.0)
    try:
        vectors: list[float] = []
        parts = space_directions.replace("(", "").split(")")
        for part in parts:
            vals = [float(x.strip()) for x in part.replace(",", " ").split() if x.strip().lower() != "none"]
            if vals:
                vectors.append(float(np.linalg.norm(vals)))
        if len(vectors) >= 3:
            return tuple(vectors[:3])  # type: ignore[return-value]
    except Exception:  # noqa: BLE001 - spacing metadata varies by exporter
        LOGGER.debug("Could not parse space directions for mesh spacing.", exc_info=True)
    return (1.0, 1.0, 1.0)


def export_mesh_stl_ply(
    mask: np.ndarray,
    header: dict[str, str],
    out_stl: Path,
    out_ply: Path,
    *,
    solid_name: str = "candidate_mask",
) -> dict[str, Any]:
    """Export ASCII STL and PLY surfaces from a binary mask."""
    if mask.max() <= 0:
        return {"mesh_export_warning": "Mask is empty; mesh export skipped."}
    try:
        spacing = spacing_from_nrrd_header(header)
        padded = np.pad(mask.astype(np.float32), 1, mode="constant", constant_values=0)
        verts, faces, _normals, _values = measure.marching_cubes(padded, level=0.5, spacing=spacing)
        verts = verts - np.asarray(spacing)
    except Exception as exc:  # noqa: BLE001 - normalize skimage failures to metadata
        LOGGER.warning("Mesh export failed: %s", exc)
        return {"mesh_export_warning": str(exc)}

    out_stl.parent.mkdir(parents=True, exist_ok=True)
    with out_stl.open("w", encoding="utf-8") as handle:
        handle.write(f"solid {solid_name}\n")
        for tri in faces:
            v1, v2, v3 = verts[tri]
            normal = np.cross(v2 - v1, v3 - v1)
            norm = np.linalg.norm(normal)
            normal = normal / norm if norm > 0 else np.array([0.0, 0.0, 0.0])
            handle.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
            handle.write("    outer loop\n")
            handle.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
            handle.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
            handle.write(f"      vertex {v3[0]:.6e} {v3[1]:.6e} {v3[2]:.6e}\n")
            handle.write("    endloop\n")
            handle.write("  endfacet\n")
        handle.write(f"endsolid {solid_name}\n")

    with out_ply.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(verts)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write(f"element face {len(faces)}\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")
        for vertex in verts:
            handle.write(f"{vertex[0]:.6e} {vertex[1]:.6e} {vertex[2]:.6e}\n")
        for tri in faces:
            handle.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")

    return {"vertices": int(len(verts)), "faces": int(len(faces)), "spacing": spacing}
