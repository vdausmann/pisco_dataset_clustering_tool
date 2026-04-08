from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .pipeline import run_pipeline, export_ecotaxa_zips, count_export_particles


def discover_recent_runs(anchor_output: str, limit: int = 20) -> List[Dict[str, str]]:
    """Find recent run directories containing at least sample_index.csv."""
    roots: List[str] = []
    if anchor_output:
        roots.append(str(Path(anchor_output).expanduser().resolve().parent))
    roots.extend(["/tmp", str(Path.cwd())])

    seen_roots = set()
    unique_roots: List[str] = []
    for root in roots:
        if root not in seen_roots:
            unique_roots.append(root)
            seen_roots.add(root)

    candidates: Dict[str, float] = {}
    for root in unique_roots:
        try:
            entries = list(Path(root).iterdir())
        except Exception:
            continue

        for entry in entries:
            if not entry.is_dir():
                continue

            try:
                sample_csv = entry / "sample_index.csv"
                if not sample_csv.exists():
                    continue
            except Exception:
                continue

            ts_candidates = [sample_csv]
            for name in ["cluster_assignments.csv", "umap_2d.csv", "features_nonliving.h5"]:
                p = entry / name
                try:
                    if p.exists():
                        ts_candidates.append(p)
                except Exception:
                    pass
            try:
                ts = max(p.stat().st_mtime for p in ts_candidates)
            except Exception:
                continue
            candidates[str(entry)] = max(candidates.get(str(entry), 0.0), ts)

    items = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    out: List[Dict[str, str]] = []
    for path, ts in items:
        has_cluster = os.path.isfile(os.path.join(path, "cluster_assignments.csv"))
        has_feats = os.path.isfile(os.path.join(path, "features_nonliving.h5"))
        stamp = pd.to_datetime(ts, unit="s").strftime("%Y-%m-%d %H:%M")
        tags = []
        if has_cluster:
            tags.append("clustered")
        if has_feats:
            tags.append("features")
        tag_str = ", ".join(tags) if tags else "sample"
        out.append(
            {
                "label": f"{Path(path).name}  [{stamp}]  ({tag_str})",
                "value": path,
            }
        )
    return out
