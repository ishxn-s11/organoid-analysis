# 🔬 Organoid Analysis

3D organoid and spheroid image analysis — from raw confocal stacks to single-cell morphology, spatial topology, and statistical comparisons.

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![Cellpose](https://img.shields.io/badge/segmentation-cellpose-green)

| Raw nuclei (z-slice) | Segmented |
|:---:|:---:|
| ![](https://raw.githubusercontent.com/ishxn-s11/organoid-analysis/main//imgs/zerog_spheroid.png) | ![](https://raw.githubusercontent.com/ishxn-s11/organoid-analysis/main/imgs/zerog_segmented.png) |
## Features

- **3D segmentation** — Cellpose-based nuclei and whole-cell detection on multi-channel confocal stacks
- **Morphological profiling** — volume, sphericity, prolate/oblate shape ratios, and statistical testing (Mann-Whitney, t-test, Cohen's d) across conditions
- **Spatial topology** — 17 neighbor-distance and density descriptors used to classify culture geometry (cup, well, cyst, spheroid)
- **Heterogeneity mapping** — unsupervised clustering (K-Means, PCA) to discover distinct cellular subpopulations within a single organoid
- **Data exploration toolkit** — automated quality checks, outlier detection, and correlation analysis for any new dataset

## Quick Start

```bash
pip install cellpose torch tifffile numpy pandas scikit-image scipy
```

```python
import tifffile as tiff
from cellpose import models

# Load a nuclei channel (single z-slice or 3D stack)
nuclei = tiff.imread("nuclei.tif")

# Segment
model = models.CellposeModel(gpu=True, model_type="nuclei")
masks, flows = model.eval(nuclei, diameter=30, channels=[0, 0])[:2]

print(f"Detected {masks.max()} nuclei")
```

## Configuration

Core parameters used across the pipeline (`organoid_segmentation.py`):

| Parameter | Nuclei | Cells |
|---|---|---|
| `model_type` | `nuclei` | `cyto3` |
| `diameter` | 30 px | 50 px |
| `flow_threshold` | 0.4 | 0.6 |
| `cellprob_threshold` | 0.0 | 0.0 |
| `pix_size_xy` | 0.345 µm | — |
| `pix_size_z` | 1.0 µm | — |

**Pipeline notebooks:**

| Notebook | Purpose |
|---|---|
| `organoid_segmentation.py` | Load stacks → normalize → Cellpose 2D/3D segmentation |
| `organoid_data_xploration.ipynb` | Dataset QC, outlier detection, correlation analysis |
| `morphological_analysis.ipynb` | Shape/volume comparison across experimental conditions |
| `spatial_topology_analysis.ipynb` | Classify culture geometry from spatial neighbor features |
| `organoid_spatial_herterogeneity.ipynb` | Cluster subpopulations within a single organoid |
