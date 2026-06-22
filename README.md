# dermal-morphometry

Reproducible Python package for SR-PhC-microCT quantification of dermal fibrous matrix remodeling in rat wound-bed ROIs, as described in:

> **Synchrotron X-ray Phase-Contrast Microtomography of Dermal Fibrous Matrix Remodeling in Rat Wounds Treated with Chitosan-Based Films**  
> Maryam Mitra Elmi, Fatemeh Elmi, Elena Longo, Armita Hoda, Amirreza Farnam Taheri, Guiliana Tromba

## Citation

Cite the associated study and this software using [`CITATION.cff`](CITATION.cff) or [`references.bib`](references.bib).

```text
Elmi MM, Elmi F, Longo E, Hoda A, Taheri AF, Tromba G.
Synchrotron X-ray Phase-Contrast Microtomography of Dermal Fibrous Matrix Remodeling
in Rat Wounds Treated with Chitosan-Based Films. [Journal, year, DOI — upon publication.]

Software: dermal-morphometry v1.0.0.
```

## Code availability

Custom analysis code supporting this study is provided as **dermal-morphometry** (v1.0.0). The package implements two SR-PhC-microCT workflows (Section 2.3): (i) low-resolution P86 fibrous-candidate extraction with connectivity metrics and (ii) high-resolution fixed-angle orientation-class mapping within wound-bed ROIs. Source code, installation instructions, and a synthetic smoke test are included in the supplementary code bundle. Analysis in the study used Python 3.13.5; the package requires Python ≥ 3.11 with NumPy, SciPy, scikit-image, and Matplotlib. *[Insert repository DOI or URL upon deposition.]*

## Authors and affiliations

| Author | Affiliation |
| --- | --- |
| Maryam Mitra Elmi | Research Center of Cellular and Molecular Biology, Health Research Center, Babol University of Medical Sciences, Babol, Iran |
| Fatemeh Elmi\* | Department of Marine Chemistry, Faculty of Marine & Environmental Sciences, University of Mazandaran, Babolsar, Iran |
| Elena Longo | Elettra Sincrotrone Trieste, S.S. 14 km 163.5, 34149 Basovizza, Trieste, Italy |
| Armita Hoda | Department of Biotechnology, College of Science, University of Tehran, Tehran, Iran |
| Amirreza Farnam Taheri | Department of Economics, Tehran Institute for Advanced Studies, Tehran, Iran |
| Guiliana Tromba | Elettra Sincrotrone Trieste, S.S. 14 km 163.5, 34149 Basovizza, Trieste, Italy |

**Corresponding author:** Fatemeh Elmi — f.elmi@umz.ac.ir, telefax +98 1135305113

## Workflows

| Analysis track | CLI mode | Manuscript role |
| --- | --- | --- |
| Low-resolution P86 fibrous-candidate extraction | `low-res` | Wound-bed organization, connectivity, fragmentation (Tables 1–2) |
| High-resolution fixed-angle orientation mapping | `high-res` | Fibrous-candidate abundance and orientation descriptors (Tables 3–5) |

## Installation

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python tests/smoke_test.py
python examples/run_synthetic_demo.py
```

**Requirements:** Python ≥ 3.11, NumPy, SciPy, scikit-image, Matplotlib. ROI cropping used 3D Slicer 5.10.2.

## Usage

```bash
# Low-resolution P86 workflow
dermal-morphometry --mode low-res --input ROI_low.nrrd --outdir results_low -v

# High-resolution orientation workflow
dermal-morphometry --mode high-res --input ROI_high.nrrd --outdir results_high --sample chitosan_G90_ROI1 -v

# Both tracks (verification only; normally use separate acquisitions)
dermal-morphometry --mode both --input ROI.nrrd --outdir results_both --sample sample_ROI -v
```

See [`examples/README.md`](examples/README.md) for a self-contained synthetic demo.

## Package layout

```
dermal_morphometry/
├── cli.py           # Command-line entry point
├── constants.py     # Normalization defaults and scientific labels
├── export.py        # CSV, JSON, README, and ZIP writers
├── mesh_io.py       # Optional STL/PLY export (low-res)
├── preprocessing.py # Normalization and downsampling
├── nrrd_io.py       # Attached-data NRRD reader/writer
├── visualization.py # QC dashboards
├── low_res.py       # P86 workflow
└── high_res.py      # Orientation-class workflow
```

## Methods defaults (Section 2.3)

**Shared:** 1st–99th percentile clipping; rescale to [0, 1].

**Low-resolution:** Gaussian σ = 1.0; P86 threshold; 3D closing (ball radius 1); remove components < 120 voxels; hole filling; density map σ = 5.0; pixel size 3.84 µm.

**High-resolution:** downsample ×6; exclude below 7th percentile; intensity P86 + texture P70 (weight 0.35); min object 8 voxels; structure tensor σ_gradient = 1.0, σ_tensor = 2.0; longitudinal 0°/180° ± 30°, transverse 90° ± 30°; pixel size 0.9 µm.

Volumes were analyzed without rotation; orientation classes are defined relative to reconstruction image axes.

## Outputs

**Low-resolution:** P86 mask, enhanced signal, skeleton QC map, density map, metrics CSV, metadata JSON, QC dashboard, MIP overlay, optional STL/PLY meshes, per-run README, ZIP.

**High-resolution:** valid-dermis mask, fibrous-candidate mask, orientation labels, angle/coherence volumes, metrics CSV, metadata JSON, QC dashboard, per-run README, ZIP.

## Data availability

SR-PhC-microCT data were acquired at the SYRMEP beamline, Elettra Sincrotrone Trieste, Italy. Raw and full reconstructed volumes may be subject to beamline and institutional policies. Processed ROI NRRD files sufficient to reproduce reported metrics are available from the corresponding author on reasonable request.

## Interpretation limits

Segmented structures are **fibrous or collagen-rich candidates**, not chemically definitive collagen, unless validated independently (staining, FTIR, SHG, histology, etc.).

Longitudinal and transverse labels are **fixed reconstruction-coordinate descriptors**, not verified in-vivo anatomical directions.

## Software note (Supplementary Methods)

Quantitative morphometry was performed on NRRD volumes exported from 3D Slicer after manual wound-bed ROI cropping. The low-resolution workflow applies P86 thresholding with morphological refinement and connectivity metrics. The high-resolution workflow combines intensity–texture candidate extraction with slice-wise 2D structure-tensor orientation classification. No proprietary dependencies are required; NRRD I/O is handled internally.

**Author contributions (software):** M.M.E. and F.E. developed the analysis workflows; F.E. supervised analysis and prepared the reproducibility package; E.L. and G.T. supported synchrotron acquisition; A.H. and A.F.T. contributed to sample preparation and coordination.

## License

Supplied for manuscript review and research reproduction. See [`LICENSE.txt`](LICENSE.txt).
