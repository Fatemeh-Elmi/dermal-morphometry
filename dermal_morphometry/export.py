"""Shared artifact writers for pipeline outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
import zipfile


def write_metrics_csv(path: Path, metrics: dict[str, Any]) -> None:
    """Write a single-row metrics table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def write_json_metadata(path: Path, metadata: dict[str, Any]) -> None:
    """Write indented JSON run metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def zip_output_directory(outdir: Path, zip_path: Path | None = None) -> Path:
    """Archive all files in an output directory into a sibling ZIP file."""
    archive_path = outdir.with_suffix(".zip") if zip_path is None else zip_path
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(outdir.iterdir()):
            if item.is_file():
                archive.write(item, arcname=f"{outdir.name}/{item.name}")
    return archive_path


def write_text_readme(path: Path, text: str) -> None:
    """Write a plain-text method summary for one run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def safe_output_stem(text: str) -> str:
    """Return a filesystem-safe sample label for output filenames."""
    return text.replace(" ", "_").replace("/", "_").replace("\\", "_")
