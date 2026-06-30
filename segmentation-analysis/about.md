# Organoid Segementation Analysis

Segment confocal z-stacks into per-cell features, then explore them in an interactive 3D dashboard.

`Python` `Dash` `Cellpose`

## Features

- **Z-stack segmentation** — nuclei + cytoplasm masks via Cellpose, with anisotropy correction for non-cubic voxels
- **Per-cell feature extraction** — volume, intensity, morphology, and spatial-neighbor descriptors
- **Interactive 3D dashboard** — live filtering, feature focus, and an orbiting Scatter3d view of the feature landscape
- **Notebook analyses** — spatial topology, intra-organoid heterogeneity, and osmotic-stress morphology studies

## Quick Start

### 1. Segmentation (z-stack → features)

```bash
pip install cellpose tifffile torch scikit-image scipy numpy pandas
```

```python
from cellpose import models
import tifffile as tiff

nuclei = tiff.imread("PDAC-C1.tiff")   # (z, y, x) stack
cells  = tiff.imread("PDAC-C2.tiff")

model = models.CellposeModel(gpu=True, model_type="cyto3")
masks, flows = model.eval(cells, diameter=50, channels=[1, 2], do_3D=False)[:2]
```

Full pipeline (normalization, 2D test slice, overlays): `organoid_segmentation.py`

### 2. Dashboard (features → visualization)

```bash
pip install -r requirements.txt
python app.py
```

Open `http://localhost:8050`.

## Configuration

| Setting | Where | Default |
|---|---|---|
| Pixel size (xy / z, μm) | `organoid_segmentation.py` | `0.345` / `1.0` |
| Nuclei diameter (px) | `organoid_segmentation.py` | `30` |
| Cell diameter (px) | `organoid_segmentation.py` | `50` |
| Flow / cellprob thresholds | `organoid_segmentation.py` | `0.4` / `0.0` (nuclei), `0.6` / `0.0` (cells) |
| Dashboard port | `app.py` | `8050` |
| Default feature column | `callbacks.py` | `cell_volume_um3` |

## Project Layout

```
organoid_segmentation.py   # Cellpose pipeline: tiff → masks → features
app.py / layout.py         # Dash app entry + UI
callbacks.py / plots.py    # Reactive logic + figure builders
utils.py / styles.py       # Dataset loader + theme tokens
*.ipynb                    # Downstream analyses (topology, heterogeneity, morphology)
```

## Data

Sample z-stacks (`*.tiff`) are 512×512, single-channel, 15–109 slices — HCT116 monolayer, colon organoids, PDAC, and breast cancer spheroid models.
