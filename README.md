# pisco_dataset_clustering_tool

Reusable Python package for PISCO validation dataset creation and non-living clustering.

## Install (editable)

```bash
pip install -e .
```

## Use in Dash apps

```python
from pisco_dataset_clustering_tool import (
    run_pipeline,
    export_ecotaxa_zips,
    count_export_particles,
    discover_recent_runs,
)
```

## Main API

- `run_pipeline(...)`: end-to-end sampling, SSL features, UMAP, clustering, export
- `export_ecotaxa_zips(...)`: filtered/split EcoTaxa export
- `count_export_particles(...)`: pre-export count preview
- `discover_recent_runs(anchor_output, limit=20)`: run directory discovery for UI dropdowns

## Notes

- Core implementation lives in `pisco_dataset_clustering_tool/pipeline.py`.
- SSL utilities live in `pisco_dataset_clustering_tool/ssl_trainer.py`.
