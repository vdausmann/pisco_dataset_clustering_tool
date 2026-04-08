# pisco_dataset_clustering_tool

Reusable package for **PISCO validation dataset creation + non-living clustering**, including a Dash control app.

## What comes before this tool

This repository assumes upstream processing already produced profile-level outputs (for example from segmentation/classification pipelines) including:

- metadata tables such as `*_crops_metadata.csv` or `*_ecotaxa.tsv`
- crop image paths (typically `Crops` / `Deconv_crops`)

The main entrypoint (`run_pipeline`) aggregates these profile outputs and builds a clustering-ready validation dataset.

## Where data comes from

- **Input root**: a directory containing processed profile folders
- **Discovered metadata**: per-profile files under profile result folders
- **Images**: file paths referenced by metadata rows

## What the product is

The product is a reproducible run directory with:

- `sample_index.csv` (sampled particle metadata)
- `features_nonliving.h5` (feature vectors)
- `umap_2d.csv` (embedding projection)
- `cluster_assignments.csv` (cluster labels)
- EcoTaxa export ZIPs (optional/manual export path)

## Pipeline workflow

1. Collect metadata across processed profiles
2. Draw sample set
3. Identify non-living subset
4. Train/load SSL model
5. Extract features
6. UMAP + clustering
7. Write cluster outputs
8. Export EcoTaxa bundles

## Install

Core package only:

```bash
pip install -e .
```

Package + Dash app dependencies:

```bash
pip install -e .[app]
```

## Use as library

```python
from pisco_dataset_clustering_tool import (
    run_pipeline,
    export_ecotaxa_zips,
    count_export_particles,
    discover_recent_runs,
)
```

## Run included Dash app

```bash
python apps/pipeline_app.py --host 0.0.0.0 --port 8060
```

## Repository layout

- `pisco_dataset_clustering_tool/pipeline.py`: core dataset + clustering pipeline
- `pisco_dataset_clustering_tool/ssl_trainer.py`: SSL training/feature extraction
- `pisco_dataset_clustering_tool/api.py`: stable reusable API
- `apps/pipeline_app.py`: Dash control + explorer app
