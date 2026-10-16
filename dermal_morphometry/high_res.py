"""High-resolution SR-PhC-microCT candidate segmentation and image orientation.

Fixed analysis settings:
- Original ROI resolution (DS1), NRRD axis order X,Y,Z.
- P1/P99 grayscale normalization, valid intensity > ROI P7.
- Candidate mask = valid AND (intensity >= valid P60 OR
  local 3D 3x3x3 standard deviation >= valid P55).
- Per-slice 2D structure tensor in YZ image-coordinate planes (sliced along X),
  Gaussian gradient sigma=1 and tensor sigma=2 voxel, class tolerance +/-30 deg.
- In-plane row axis Y and column axis Z; class labels refer to image axes
  and are not registered to an anatomical wound axis.
- Fraction denominator: all ROI voxels, including excluded background.

Requires Python >=3.10, numpy, scipy; matplotlib optional for --qc.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import re
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy import ndimage as ndi

PROFILES = {
    "p60p55-ds1": {
        "downsample": 1,
        "plane": "YZ",
        "sigma_gradient": 1.0,
        "sigma_tensor": 2.0,
        "status": "YZ image-coordinate analysis (row Y, column Z)",
    },
}

DTYPES = {
    "unsigned char": "u1", "uchar": "u1", "uint8": "u1",
    "signed char": "i1", "char": "i1", "int8": "i1",
    "unsigned short": "u2", "ushort": "u2", "uint16": "u2",
    "short": "i2", "int16": "i2", "int": "i4", "int32": "i4",
    "float": "f4", "double": "f8",
}


def read_nrrd(path: Path) -> tuple[np.ndarray, dict[str, str]]:
    """Read single-file 3-D NRRD preserving its axis order (axis0,axis1,axis2).

    Does not secretly transpose X,Y,Z to Z,Y,X. Supports gzip and raw encoding.
    """
    with path.open("rb") as f:
        payload = f.read()
    match = re.search(rb"\r?\n\r?\n", payload)
    if match is None:
        raise ValueError(f"Invalid NRRD: no header separator in {path}")
    head, body = payload[: match.start()], payload[match.end():]
    if not head.startswith(b"NRRD"):
        raise ValueError(f"Not a NRRD: {path}")
    header: dict[str, str] = {}
    for row in head.decode("latin1").splitlines():
        if row.startswith("#") or ":" not in row:
            continue
        key, value = row.split(":", 1)
        header[key.strip().lower()] = value.strip()
    if int(header.get("dimension", "0")) != 3:
        raise ValueError("Only scalar 3D NRRDs are supported")
    if "data file" in header or "datafile" in header:
        raise ValueError("Detached NRRD data is not supported")
    dtype_code = DTYPES.get(header.get("type", "").lower())
    if dtype_code is None:
        raise ValueError("Unsupported NRRD scalar datatype")
    dtype = np.dtype(dtype_code)
    if dtype.itemsize > 1:
        endian = header.get("endian", "little").lower()
        if endian not in ("little", "big"):
            raise ValueError("Invalid NRRD endian")
        dtype = dtype.newbyteorder("<" if endian == "little" else ">")
    encoding = header.get("encoding", "raw").lower()
    if encoding in ("gzip", "gz"):
        body = gzip.decompress(body)
    elif encoding != "raw":
        raise ValueError(f"NRRD encoding not supported: {encoding}")
    dims = tuple(int(s) for s in header["sizes"].split())
    arr = np.frombuffer(body, dtype=dtype)
    if arr.size != int(np.prod(dims)):
        raise ValueError(f"NRRD voxel count mismatch: got {arr.size}, expect {dims}")
    return np.array(arr.reshape(dims, order="F"), copy=True), header


def verify_xyz_header(header: dict[str, str]) -> None:
    """Verify that the expected physical X,Y,Z NRRD axis order is present."""
    directions = header.get("space directions", "")
    triples = re.findall(r"\(([^)]*)\)", directions)
    if len(triples) != 3:
        raise ValueError("Three NRRD 'space directions' vectors are required")
    v = np.array([[float(x) for x in tri.split(",")] for tri in triples])
    if v.shape != (3, 3) or not np.all(np.isfinite(v)):
        raise ValueError("Invalid space directions")
    if not (abs(v[0, 0]) > 0 and abs(v[1, 1]) > 0 and abs(v[2, 2]) > 0):
        raise ValueError("Expected axis0=X, axis1=Y, axis2=Z")
    if not np.allclose(v - np.diag(np.diag(v)), 0.0, atol=1e-6):
        raise ValueError("Non-axis-aligned NRRD; must handle full geometry first")


def normalize_and_segment(vol_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Precisely reproduces audited P60/P55 segmentation, no morphology."""
    lo, hi = (float(x) for x in np.percentile(vol_raw, [1, 99]))
    if hi <= lo:
        raise ValueError("Cannot normalize constant-intensity volume")
    vol = np.clip((vol_raw.astype("f4") - lo) / (hi - lo), 0, 1).astype("f4")
    p7 = float(np.percentile(vol, 7))
    valid = vol > p7
    if not valid.any():
        raise ValueError("Valid tissue mask empty")
    # Reflected 3x3x3 population standard deviation: not gradient magnitude.
    x = vol.astype("f8")
    mean = ndi.uniform_filter(x, size=3, mode="reflect")
    std = np.sqrt(np.maximum(ndi.uniform_filter(x * x, size=3, mode="reflect") - mean**2, 0.0))
    p60 = float(np.percentile(vol[valid], 60))
    p55 = float(np.percentile(std[valid], 55))
    fibrous = valid & ((vol >= p60) | (std >= p55))
    return vol, valid, fibrous, {"norm_P1": lo, "norm_P99": hi,
                                 "valid_P7_normalized": p7, "intensity_P60_valid": p60,
                                 "local_SD_P55_valid": p55}


def orientation_and_metrics(
    vol: np.ndarray, valid: np.ndarray, fibrous: np.ndarray,
    *, plane: str, sigma_gradient: float, sigma_tensor: float,
    angle_tolerance_deg: float = 30.0,
) -> tuple[dict, np.ndarray]:
    """Slice-wise 2D structure tensor in image-coordinate planes.

    For the fixed YZ analysis, the volume is stored as [X,Y,Z]; slicing
    axis 0 gives image rows along Y and columns along Z. The reported row
    and column aligned classes are in-plane image-coordinate descriptors.
    Fiber orientation uses gradient principal eigenvector + 90 degrees.
    """
    if plane != "YZ":
        raise ValueError("Fixed analysis uses YZ in-plane slices")
    perm = (0, 1, 2)  # [X,Y,Z] => row:Y, column:Z
    a = vol.transpose(perm)
    f = fibrous.transpose(perm)
    v = valid.transpose(perm)
    labels = np.zeros(a.shape, dtype=np.uint8)
    # classes 0=excluded, 1=valid noncandidate, 2=oblique,
    # 3=column-axis aligned, 4=row-axis aligned
    counts = np.zeros(3, dtype=np.int64)  # oblique, row-aligned, column-aligned
    coherence_sum = 0.0
    for i in range(a.shape[0]):
        img = a[i]
        smooth = ndi.gaussian_filter(img, sigma=sigma_gradient)
        g_row, g_col = np.gradient(smooth)
        cc = ndi.gaussian_filter(g_col * g_col, sigma=sigma_tensor)
        rr = ndi.gaussian_filter(g_row * g_row, sigma=sigma_tensor)
        cr = ndi.gaussian_filter(g_col * g_row, sigma=sigma_tensor)
        theta = (np.degrees(0.5 * np.arctan2(2 * cr, cc - rr) + np.pi / 2) % 180)
        coh = np.clip(np.sqrt((cc - rr)**2 + 4 * cr * cr) / (cc + rr + 1e-8), 0, 1)
        fm = f[i]
        col_aligned = fm & ((theta <= angle_tolerance_deg) | (theta >= 180 - angle_tolerance_deg))
        row_aligned = fm & (np.abs(theta - 90) <= angle_tolerance_deg)
        oblique = fm & ~(col_aligned | row_aligned)
        labels[i][v[i]] = 1
        labels[i][oblique] = 2
        labels[i][col_aligned] = 3
        labels[i][row_aligned] = 4
        counts += np.array([int(oblique.sum()), int(row_aligned.sum()), int(col_aligned.sum())], dtype=np.int64)
        coherence_sum += float(coh[fm].sum(dtype="f8"))
    out_labels = np.transpose(labels, np.argsort(perm)).copy()
    N = vol.size
    n_fiber = int(fibrous.sum())
    n_ob, n_row, n_col = map(int, counts)
    if n_ob + n_row + n_col != n_fiber:
        raise AssertionError("Candidate mask / orientation labels must agree")
    mean_coh = coherence_sum / n_fiber if n_fiber else float("nan")
    return ({
        "N_ROI": int(N), "N_valid": int(valid.sum()), "N_fibrous": n_fiber,
        "N_oblique": n_ob, "N_row_aligned": n_row, "N_column_aligned": n_col,
        "valid_percent": float(100 * valid.sum() / N),
        "fibrous_candidate_percent": float(100 * n_fiber / N),
        "oblique_percent": float(100 * n_ob / N),
        "row_axis_percent": float(100 * n_row / N),
        "column_axis_percent": float(100 * n_col / N),
        "row_over_column_ratio": float(n_row / n_col) if n_col else float("nan"),
        "mean_coherence_fibrous": float(mean_coh),
        "alignment_imbalance_index": float(abs(n_row - n_col) / n_fiber) if n_fiber else float("nan"),
    }, out_labels)


def write_uint8_nrrd(path: Path, data_xyz: np.ndarray, header: dict[str, str], stride: int) -> None:
    """Export labels/masks preserving original voxel axis order and geometry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dirs = header.get("space directions", "")
    triples = re.findall(r"\(([^)]*)\)", dirs)
    if len(triples) == 3:
        vectors = [[float(x) * stride for x in tri.split(",")] for tri in triples]
        directions = " ".join("(" + ",".join(f"{v:g}" for v in vec) + ")" for vec in vectors)
    else:
        directions = None
    lines = ["NRRD0004", "type: unsigned char", "dimension: 3",
             "sizes: " + " ".join(str(n) for n in data_xyz.shape),
             "encoding: gzip", "endian: little"]
    if "space" in header:
        lines.append("space: " + header["space"])
    if directions is not None:
        lines.append("space directions: " + directions)
    if "space origin" in header:
        lines.append("space origin: " + header["space origin"])
    lines.append("kinds: domain domain domain")
    payload = ("\n".join(lines) + "\n\n").encode("utf8") + gzip.compress(data_xyz.astype(np.uint8).tobytes(order="F"))
    path.write_bytes(payload)


def save_qc(path: Path, vol_xyz: np.ndarray, labels_xyz: np.ndarray,
            plane: str, name: str, mode: str) -> None:
    """The image is generated from real voxels; NO interpolation of labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    a = vol_xyz
    b = labels_xyz
    mids = [max(0, round(a.shape[0] * p)) for p in (0.2, 0.5, 0.8)]
    mids = [min(k, a.shape[0] - 1) for k in mids]
    cmap = ListedColormap(["#111111", "#bfe9e8", "#a4b5a3", "#e8de53", "#295fea"])
    norm = BoundaryNorm(np.arange(-0.5, 5.5), 5)
    fig, axs = plt.subplots(2, 3, figsize=(11.5, 7), dpi=160)
    for j, k in enumerate(mids):
        axs[0, j].imshow(a[k], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axs[1, j].imshow(b[k], cmap=cmap, norm=norm, interpolation="nearest")
        axs[0, j].set_title(f"Real NRRD slice {k} | {plane}", fontsize=10)
        axs[1, j].set_title(f"Orientation classes | slice {k}", fontsize=10)
        for ax in (axs[0, j], axs[1, j]):
            ax.set_axis_off()
    fig.suptitle(f"{name} | {mode} | {plane} image-coordinate plane", fontsize=13)
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=label) for c, label in zip(
        ["#111111", "#bfe9e8", "#a4b5a3", "#e8de53", "#295fea"],
        ["excluded", "valid background", "oblique", "column-axis aligned", "row-axis aligned"])]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, 0.015), frameon=False)
    fig.subplots_adjust(bottom=0.12, top=0.88, hspace=0.20, wspace=0.07)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def analyze(path: Path, outdir: Path, profile: str, *, sample: str = "",
            qc: bool = False, save_labels: bool = False) -> dict:
    settings = PROFILES[profile]
    outdir.mkdir(parents=True, exist_ok=True)
    original, header = read_nrrd(path)
    verify_xyz_header(header)  # Required for this study's axis-specific analysis.
    ds = int(settings["downsample"])
    raw = original[::ds, ::ds, ::ds]
    normalized, valid, fibrous, thresholds = normalize_and_segment(raw)
    stats, labels = orientation_and_metrics(normalized, valid, fibrous,
            plane=str(settings["plane"]),
            sigma_gradient=float(settings["sigma_gradient"]),
            sigma_tensor=float(settings["sigma_tensor"]))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    row = {
        "sample": sample or path.stem, "input_file": path.name, "input_sha256": sha,
        "profile": profile, "validation_status": settings["status"],
        "NRRD_axis_order": "X,Y,Z", "slice_plane": settings["plane"],
        "downsample_factor": ds, "sigma_gradient": settings["sigma_gradient"],
        "sigma_tensor": settings["sigma_tensor"], "tolerance_degrees": 30,
        "axis_row_in_slice": "Y",
        "axis_column_in_slice": "Z",
        "original_shape_XYZ": "x".join(map(str, original.shape)),
        "analyzed_shape_XYZ": "x".join(map(str, raw.shape)),
        **thresholds, **stats,
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
    }
    with (outdir / "metrics.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys()); writer.writeheader(); writer.writerow(row)
    (outdir / "provenance.json").write_text(json.dumps(row, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")
    if save_labels:
        write_uint8_nrrd(outdir / "orientation_classes_XYZ.nrrd", labels, header, ds)
        write_uint8_nrrd(outdir / "fibrous_candidate_mask_XYZ.nrrd", fibrous, header, ds)
        write_uint8_nrrd(outdir / "valid_tissue_mask_XYZ.nrrd", valid, header, ds)
    if qc:
        save_qc(outdir / "QC_real_voxels.png", normalized, labels,
                str(settings["plane"]), sample or path.stem, profile)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="Single-file NRRD source ROI")
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--sample", default="")
    parser.add_argument("--profile", choices=list(PROFILES), default="p60p55-ds1")
    parser.add_argument("--qc", action="store_true", help="Generate actual-voxel QC PNG (requires matplotlib)")
    parser.add_argument("--save-labels", action="store_true", help="Write masks and labels as geometry-preserving NRRDs")
    args = parser.parse_args()
    result = analyze(args.input, args.outdir, args.profile,
                     sample=args.sample, qc=args.qc, save_labels=args.save_labels)
    print(json.dumps({key: result[key] for key in (
        "profile", "fibrous_candidate_percent", "oblique_percent",
        "row_axis_percent", "column_axis_percent", "row_over_column_ratio",
        "mean_coherence_fibrous")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
