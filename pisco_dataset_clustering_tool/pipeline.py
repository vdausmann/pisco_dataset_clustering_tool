#!/usr/bin/env python3
"""
Create a validation dataset from processed PISCO profiles.

Pipeline
--------
1. Collect all *_crops_metadata.csv files produced by process_pisco_profiles.py
2. Draw a uniform random sample of N images (default 100 000)
3. Identify non-living crops (ViT top-1 class contains 'non_living' or 'detritus')
4. Train a SimCLR model on ALL non-living crops found in the full collection
   (or load an existing checkpoint via --ssl-model) — skipped with --no-train
5. Extract 2048-d features for non-living crops in the sample  →  features.h5
6. UMAP (64-D cosine) → HDBSCAN clustering
7. Assign every non-living crop (incl. noise points) to best cluster
8. Export EcoTaxa-compatible ZIP(s) split at --max-zip-mb (default 500 MB)
   - Non-living crops:  object_annotation_category = 'non_living/cluster_NN'
   - Living crops:      keep original ViT top-1 prediction

Usage examples
--------------
# Full run (train SimCLR + cluster + export):
  python create_validation_dataset.py \\
      --source /mnt/filer \\
      --output /mnt/filer/validation_dataset

# Reuse an existing SSL model:
  python create_validation_dataset.py \\
      --source /mnt/filer \\
      --output /mnt/filer/validation_dataset \\
      --ssl-model models/ssl_nonliving.pth

# Dry run — 1000 samples, no training:
  python create_validation_dataset.py \\
      --source /mnt/filer \\
      --output /tmp/dry_run \\
      --n-samples 1000 \\
      --no-train \\
      --ssl-model models/ssl_nonliving.pth
"""

import os
import re
import shutil
import zipfile
import logging
import argparse
import datetime
import json
from glob import glob
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import h5py
import umap
from sklearn.cluster import HDBSCAN, KMeans

# Local module (same directory)
try:
    from .ssl_trainer import train_ssl_model, extract_features, extract_features_imagenet
except ImportError:
    from ssl_trainer import train_ssl_model, extract_features, extract_features_imagenet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Collect metadata across all processed profiles
# ---------------------------------------------------------------------------

def collect_all_crops(
    source_root: str,
    metadata_patterns: Optional[Tuple[str, ...]] = None,
    metadata_choice: str = "auto",
    crop_path_token: Optional[str] = "Deconv_crops",
    crop_filter_strict: bool = True,
    rewrite_crops_to_deconv: bool = True,
) -> pd.DataFrame:
    """
        Collect metadata by traversing profile folders directly (fast):
            <profiles_base>/<profile>/<profile>_Results/EcoTaxa/<metadata file>

        This avoids expensive recursive glob scans over the full source tree.
    """
    if metadata_patterns is None:
        metadata_patterns = (
            "*_crops_metadata.csv",
            "*_metadata.csv",
            "ecotaxa_metadata.tsv",
            "*_ecotaxa.tsv",
        )

    csv_files = _find_metadata_files_fast(
        source_root=source_root,
        metadata_patterns=metadata_patterns,
        metadata_choice=metadata_choice,
    )

    if not csv_files:
        searched = ", ".join(metadata_patterns)
        raise FileNotFoundError(
            f"No metadata files found under {source_root}. "
            f"Searched patterns: {searched}"
        )

    log.info(f"Found {len(csv_files)} metadata files")

    dfs: List[pd.DataFrame] = []
    for csv_path in sorted(csv_files):
        try:
            sep = "\t" if csv_path.lower().endswith(".tsv") else ","
            df = pd.read_csv(csv_path, low_memory=False, sep=sep)
            if sep == "\t":
                df = _drop_ecotaxa_dtype_row(df)
            df["_source_csv"] = csv_path
            dfs.append(df)
        except Exception as exc:
            log.warning(f"  Skipped {csv_path}: {exc}")

    if not dfs:
        raise RuntimeError("All metadata CSVs were empty or unreadable.")

    combined = pd.concat(dfs, ignore_index=True)
    log.info(f"Total crops collected: {len(combined):,}")

    # Deduplicate on full_path (the reliable primary key across profiles)
    path_col = _image_path_col(combined)
    before = len(combined)
    combined = combined.drop_duplicates(subset=path_col, keep="first")
    removed = before - len(combined)
    if removed:
        log.info(f"Removed {removed:,} duplicate entries (same full_path)")

    # Drop rows without a usable image path
    mask = combined[path_col].apply(lambda p: isinstance(p, str) and len(p) > 0)
    combined = combined[mask].reset_index(drop=True)
    log.info(f"Crops with valid image paths: {len(combined):,}")

    # Rewrite _Results/Crops/<file> -> _Results/Deconv_crops/<file>
    # so the same crop filename points to deconvolved imagery.
    if rewrite_crops_to_deconv:
        original_paths = combined[path_col].astype(str)
        rewritten_paths = original_paths.str.replace(
            r"([/\\])Crops([/\\])",
            r"\1Deconv_crops\2",
            regex=True,
        )
        n_rewritten = int((original_paths != rewritten_paths).sum())
        combined[path_col] = rewritten_paths
        log.info(
            f"Rewrote {n_rewritten:,} path(s) from /Crops/ to /Deconv_crops/"
        )

    # Optionally restrict to a path token (e.g. Deconv_crops)
    token_raw = None if crop_path_token is None else str(crop_path_token).strip()
    if token_raw and token_raw.lower() not in {"off", "none", "no"}:
        path_values = combined[path_col].astype(str).str.lower()

        if token_raw.lower() == "auto":
            deconv_mask = path_values.str.contains("deconv_crops", na=False)
            if int(deconv_mask.sum()) > 0:
                combined = combined[deconv_mask].reset_index(drop=True)
                log.info(
                    "Path-token auto: selected Deconv_crops paths "
                    f"({len(combined):,} rows)"
                )
            else:
                crops_mask = path_values.str.contains(r"(?:^|[\\/])crops(?:[\\/])", regex=True, na=False)
                if int(crops_mask.sum()) > 0:
                    combined = combined[crops_mask].reset_index(drop=True)
                    log.info(
                        "Path-token auto: no Deconv_crops found, falling back to Crops paths "
                        f"({len(combined):,} rows)"
                    )
                else:
                    log.warning(
                        "Path-token auto: neither Deconv_crops nor Crops found in paths; "
                        "keeping all valid paths"
                    )
        else:
            token = token_raw.lower()
            token_mask = path_values.str.contains(token, na=False)
            kept = int(token_mask.sum())
            dropped = len(combined) - kept
            if kept == 0:
                msg = (
                    f"No crop rows matched --crop-path-token '{crop_path_token}'. "
                    "Check metadata paths or use --crop-path-token auto / --no-crop-path-filter."
                )
                if crop_filter_strict:
                    raise RuntimeError(msg)
                log.warning(msg + " Keeping all valid paths.")
            else:
                combined = combined[token_mask].reset_index(drop=True)
                log.info(
                    f"Path-token filter '{crop_path_token}': kept {kept:,}, dropped {dropped:,}"
                )

    # Per-cruise summary
    if "cruise" in combined.columns:
        for cruise, grp in combined.groupby("cruise"):
            log.info(f"  {cruise:30s}: {len(grp):>8,} crops")

    return combined


def _find_metadata_files_fast(
    source_root: str,
    metadata_patterns: Tuple[str, ...],
    metadata_choice: str = "auto",
) -> List[str]:
    """
    Fast metadata discovery (no recursive ** over whole tree):
      1) Resolve one or more PISCO-Profiles base directories.
      2) Iterate profile directories directly.
      3) Pick at most one metadata file from each profile's EcoTaxa folder.
    """
    if metadata_choice not in {"auto", "crops", "ecotaxa"}:
        raise ValueError("metadata_choice must be one of: auto, crops, ecotaxa")

    source_path = Path(source_root)
    profile_bases = _resolve_profile_bases(source_path)
    log.info(f"Profiles base dirs detected: {len(profile_bases)}")
    for base in profile_bases[:10]:
        log.info(f"  profiles base: {base}")

    selected: List[str] = []
    scanned_profiles = 0
    for base in profile_bases:
        for profile_name in get_available_profiles(str(base)):
            scanned_profiles += 1
            picked = _pick_profile_metadata_file(
                base_dir=base,
                profile_name=profile_name,
                metadata_choice=metadata_choice,
                metadata_patterns=metadata_patterns,
            )
            if picked is not None:
                selected.append(str(picked))

            if scanned_profiles % 1000 == 0:
                log.info(f"  scanned profiles: {scanned_profiles}")

    if selected:
        log.info(f"Selected metadata files from profiles: {len(selected)}")
        return sorted(set(selected))

    log.warning("No metadata found using profile-based scan")
    return []


def detect_profiles_dir(root_path: str, cruise: str) -> Optional[str]:
    """Auto-detect the PISCO-Profiles subdirectory inside a cruise folder."""
    cruise_dir = Path(root_path) / cruise
    if not cruise_dir.exists():
        return None
    for child in sorted(cruise_dir.iterdir()):
        if child.is_dir() and ("PISCO-Profiles" in child.name or "Profiles" in child.name):
            return str(child)
    return str(cruise_dir)


def get_available_profiles(base_dir: str) -> List[str]:
    """Return sorted list of profile folders that have a *_Results/EcoTaxa directory."""
    base = Path(base_dir)
    if not base.exists():
        return []

    profiles: List[str] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        eco_dir = child / f"{child.name}_Results" / "EcoTaxa"
        if eco_dir.exists():
            profiles.append(child.name)
    return profiles


def _resolve_profile_bases(source_path: Path) -> List[Path]:
    """
    Resolve one or more directories that directly contain profile folders.
    Supports:
      - /mnt/filer/M181
      - /mnt/filer/M181/M181-PISCO-Profiles
      - /mnt/filer   (contains cruise dirs)
    """
    bases: List[Path] = []

    if not source_path.exists():
        return bases

    # Case 1: source is already a profiles base.
    if source_path.is_dir() and ("PISCO-Profiles" in source_path.name or "Profiles" in source_path.name):
        bases.append(source_path)
        return sorted(set(bases))

    # Case 2: source is a cruise dir containing <cruise>-PISCO-Profiles.
    try:
        for child in sorted(source_path.iterdir()):
            if child.is_dir() and ("PISCO-Profiles" in child.name or "Profiles" in child.name):
                bases.append(child)
    except Exception:
        pass

    if bases:
        return sorted(set(bases))

    # Case 3: source is root containing cruise dirs.
    try:
        for cruise_dir in sorted(source_path.iterdir()):
            if not cruise_dir.is_dir():
                continue
            detected = detect_profiles_dir(str(source_path), cruise_dir.name)
            if detected is not None:
                detected_path = Path(detected)
                if detected_path.exists() and detected_path.is_dir() and detected_path != cruise_dir:
                    bases.append(detected_path)
    except Exception:
        pass

    return sorted(set(bases))


def _pick_profile_metadata_file(
    base_dir: Path,
    profile_name: str,
    metadata_choice: str,
    metadata_patterns: Tuple[str, ...],
) -> Optional[Path]:
    """Pick one metadata file for a profile based on priority and availability."""
    eco_dir = base_dir / profile_name / f"{profile_name}_Results" / "EcoTaxa"
    if not eco_dir.exists():
        return None

    # Fast exact-name checks first.
    crops_csv = eco_dir / f"{profile_name}_crops_metadata.csv"
    eco_tsv = eco_dir / f"{profile_name}_ecotaxa.tsv"
    eco_meta_tsv = eco_dir / "ecotaxa_metadata.tsv"

    if metadata_choice == "crops":
        ordered = [crops_csv, eco_meta_tsv, eco_tsv]
    elif metadata_choice == "ecotaxa":
        ordered = [eco_tsv, eco_meta_tsv, crops_csv]
    else:  # auto
        ordered = [crops_csv, eco_meta_tsv, eco_tsv]

    for candidate in ordered:
        if candidate.exists():
            return candidate

    # Fallback: try user-supplied patterns within this EcoTaxa directory only.
    try:
        names = os.listdir(eco_dir)
    except Exception:
        return None

    for patt in metadata_patterns:
        for name in names:
            if Path(name).match(patt):
                return eco_dir / name
    return None


def _drop_ecotaxa_dtype_row(df: pd.DataFrame) -> pd.DataFrame:
    """Drop first row if it looks like EcoTaxa type tags ([f], [t], ...)."""
    if df.empty:
        return df
    first = df.iloc[0].astype(str).str.strip()
    tag_ratio = first.str.match(r"^\[[A-Za-z]\]$").mean()
    if tag_ratio >= 0.6:
        return df.iloc[1:].reset_index(drop=True)
    return df


def _image_path_col(df: pd.DataFrame) -> str:
    """Return the column that holds the absolute crop image path."""
    for col in ("full_path", "object_full_path", "img_path"):
        if col in df.columns:
            return col
    raise KeyError(
        f"No recognised image-path column found. "
        f"Available columns: {list(df.columns[:15])}"
    )


def _top1_col(df: pd.DataFrame) -> str:
    """Return the ViT top-1 prediction column name."""
    for col in ("top1", "object_annotation_category"):
        if col in df.columns:
            return col
    raise KeyError("No ViT top-1 prediction column found in dataframe.")


# ---------------------------------------------------------------------------
# 2. Random sampling
# ---------------------------------------------------------------------------

def random_sample(
    df: pd.DataFrame,
    n: int = 100_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Uniform random sample of n rows (without replacement)."""
    if n >= len(df):
        log.warning(
            f"Requested {n:,} samples but only {len(df):,} crops available "
            "— using all."
        )
        return df.copy()
    sampled = df.sample(n=n, random_state=seed).reset_index(drop=True)
    log.info(f"Sampled {len(sampled):,} crops  (seed={seed})")
    return sampled


# ---------------------------------------------------------------------------
# 3. Non-living identification
# ---------------------------------------------------------------------------

def non_living_mask(
    df: pd.DataFrame,
    top1_col: str,
    classes: Tuple[str, ...],
) -> pd.Series:
    """
    Boolean Series: True where top-1 prediction matches a non-living class.

    Matching is normalised so variants like
      non_living / non-living / non living
    are treated as equivalent.
    """
    values = (
        df[top1_col]
        .astype(str)
        .str.lower()
        .str.replace(r"[_\-]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    class_tokens = []
    for item in classes:
        token = str(item).lower()
        token = token.replace("_", " ").replace("-", " ")
        token = " ".join(token.split())
        if token:
            class_tokens.append(token)

    if not class_tokens:
        return pd.Series(False, index=df.index)

    pattern = "|".join(map(lambda x: x.replace(" ", r"\s+"), class_tokens))
    return values.str.contains(pattern, case=False, na=False, regex=True)


# ---------------------------------------------------------------------------
# 4+5. Feature cache helpers (HDF5, MorphoCluster-compatible)
# ---------------------------------------------------------------------------

def save_features_h5(
    features: np.ndarray,
    paths: List[str],
    object_ids: List[str],
    h5_path: str,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(h5_path)), exist_ok=True)
    str_dt = h5py.special_dtype(vlen=str)
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("features",   data=features,                  compression="gzip")
        f.create_dataset("paths",      data=np.array(paths, dtype=object), dtype=str_dt)
        f.create_dataset("object_ids", data=np.array(object_ids, dtype=object), dtype=str_dt)
    log.info(f"Features saved → {h5_path}  (shape={features.shape})")


def load_features_h5(
    h5_path: str,
) -> Tuple[np.ndarray, List[str], List[str]]:
    with h5py.File(h5_path, "r") as f:
        features   = f["features"][:]
        paths      = [p.decode() if isinstance(p, bytes) else str(p) for p in f["paths"][:]]
        object_ids = [p.decode() if isinstance(p, bytes) else str(p) for p in f["object_ids"][:]]
    return features, paths, object_ids


def augment_features_with_object_area(
    features: np.ndarray,
    feat_oids: List[str],
    df_nonliving_sample: pd.DataFrame,
    mode: str = "off",
    weight: float = 0.0,
) -> np.ndarray:
    """
    Optionally append a weighted, normalized log(object_area) channel to features.

    mode:
      - off
      - append_log_area
    """
    mode_norm = str(mode or "off").strip().lower()
    if mode_norm == "off" or weight == 0.0:
        return features

    if mode_norm != "append_log_area":
        log.warning(f"Unknown size-aware mode '{mode}'. Skipping size feature.")
        return features

    area_col = None
    for candidate in ("object_area", "area"):
        if candidate in df_nonliving_sample.columns:
            area_col = candidate
            break
    if area_col is None:
        log.warning("Size-aware clustering requested, but no area column found. Skipping.")
        return features

    if "object_id" in df_nonliving_sample.columns:
        area_map = pd.Series(
            pd.to_numeric(df_nonliving_sample[area_col], errors="coerce").values,
            index=df_nonliving_sample["object_id"].astype(str).values,
        )
        area_values = np.array([area_map.get(str(oid), np.nan) for oid in feat_oids], dtype=float)
    else:
        area_values = pd.to_numeric(df_nonliving_sample[area_col], errors="coerce").values
        if len(area_values) != len(feat_oids):
            log.warning(
                "Size-aware clustering requested, but object_id missing and row counts mismatch. Skipping."
            )
            return features

    valid = np.isfinite(area_values) & (area_values > 0)
    if not np.any(valid):
        log.warning("Size-aware clustering requested, but no valid positive areas found. Skipping.")
        return features

    fill_value = float(np.nanmedian(area_values[valid]))
    area_values = np.where(valid, area_values, fill_value)
    size_log = np.log1p(area_values)
    mean = float(np.mean(size_log))
    std = float(np.std(size_log))
    if std <= 1e-12:
        log.warning("Area channel has near-zero variance. Skipping size-aware augmentation.")
        return features

    size_z = (size_log - mean) / std
    size_channel = (float(weight) * size_z).reshape(-1, 1).astype(np.float32)
    features_aug = np.concatenate([features.astype(np.float32), size_channel], axis=1)

    log.info(
        f"Size-aware clustering enabled: mode={mode_norm}, area_col={area_col}, "
        f"weight={weight}, valid_area={int(np.sum(valid)):,}/{len(area_values):,}, "
        f"feature_dim {features.shape[1]}→{features_aug.shape[1]}"
    )
    return features_aug


# ---------------------------------------------------------------------------
# 6. UMAP + HDBSCAN clustering
# ---------------------------------------------------------------------------

def cluster_features(
    features: np.ndarray,
    n_umap_components: int = 64,
    min_cluster_size: int = 100,
    hdbscan_min_samples: Optional[int] = None,
    hdbscan_cluster_selection_epsilon: float = 0.0,
    hdbscan_cluster_selection_method: str = "eom",
    max_clusters: Optional[int] = None,
    umap_n_neighbors: int = 30,
    umap_min_dist: float = 0.0,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, HDBSCAN, bool]:
    """
    UMAP dimensionality reduction followed by HDBSCAN clustering.

    Returns
    -------
    embedding  : np.ndarray (N, n_umap_components)
    labels     : np.ndarray (N,)  — cluster index, -1 = noise before reassignment
    hdb_model  : fitted HDBSCAN object
    all_noise  : True if HDBSCAN produced zero clusters
    """
    log.info(
        f"UMAP: {features.shape[1]}D → {n_umap_components}D  "
        f"(n_neighbors={umap_n_neighbors}, min_dist={umap_min_dist}, metric=cosine)"
    )
    reducer = umap.UMAP(
        n_components=n_umap_components,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric="cosine",
        random_state=random_state,
    )
    embedding = reducer.fit_transform(features)
    log.info("UMAP complete")

    log.info(
        f"HDBSCAN: min_cluster_size={min_cluster_size}"
    )
    hdb = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=hdbscan_min_samples,
        metric="euclidean",
        cluster_selection_epsilon=hdbscan_cluster_selection_epsilon,
        cluster_selection_method=hdbscan_cluster_selection_method,
        store_centers="medoid",   # sklearn >= 1.3
    )
    labels = hdb.fit_predict(embedding)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int(np.sum(labels == -1))
    log.info(
        f"HDBSCAN: {n_clusters} clusters, {n_noise} noise points "
        f"({100 * n_noise / max(len(labels), 1):.1f}%)"
    )
    all_noise = n_clusters == 0

    if max_clusters is not None and n_clusters > max_clusters:
        labels = merge_to_max_clusters(embedding, labels, max_clusters)

    return embedding, labels, hdb, all_noise


def kmeans_fallback_labels(
    embedding: np.ndarray,
    k: int,
    random_state: int = 42,
) -> np.ndarray:
    """Deterministic KMeans fallback labels on embedding space."""
    if k < 2:
        raise ValueError("fallback_kmeans_k must be >= 2")
    if embedding.shape[0] < k:
        raise ValueError(
            f"fallback_kmeans_k={k} is larger than number of points={embedding.shape[0]}"
        )
    log.info(f"KMeans fallback: k={k}, random_state={random_state}")
    km = KMeans(n_clusters=k, random_state=random_state, n_init=20)
    labels = km.fit_predict(embedding)
    return labels.astype(int)


# ---------------------------------------------------------------------------
# 7. Hard assignment of noise to nearest cluster centroid
# ---------------------------------------------------------------------------

def merge_to_max_clusters(
    embedding: np.ndarray,
    labels: np.ndarray,
    max_clusters: int,
) -> np.ndarray:
    """
    Iteratively merge the two closest clusters (by centroid distance in the
    UMAP embedding) until at most *max_clusters* remain.
    Noise points (label == -1) are ignored during merging.
    """
    labels = labels.copy()
    current_ids = sorted(set(labels) - {-1})
    n = len(current_ids)
    if n <= max_clusters:
        return labels

    log.info(f"Merging {n} clusters down to max {max_clusters} ...")
    while len(current_ids) > max_clusters:
        # Compute centroid for each cluster
        centroids = {cid: embedding[labels == cid].mean(axis=0) for cid in current_ids}
        # Find the pair of clusters with the closest centroids
        best_dist = float("inf")
        merge_a, merge_b = current_ids[0], current_ids[1]
        ids = current_ids
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                d = np.linalg.norm(centroids[ids[i]] - centroids[ids[j]])
                if d < best_dist:
                    best_dist = d
                    merge_a, merge_b = ids[i], ids[j]
        # Reassign all points of merge_b → merge_a
        labels[labels == merge_b] = merge_a
        current_ids.remove(merge_b)

    # Remap cluster ids to a clean 0-based sequence
    id_map = {old: new for new, old in enumerate(sorted(current_ids))}
    for old, new in id_map.items():
        labels[labels == old] = -(new + 1)   # temp negative to avoid collisions
    for new_neg in range(len(current_ids)):
        labels[labels == -(new_neg + 1)] = new_neg

    log.info(f"After merging: {len(set(labels) - {-1})} clusters")
    return labels


def assign_noise_to_nearest(
    embedding: np.ndarray,
    labels: np.ndarray,
    hdb_model: HDBSCAN,
) -> np.ndarray:
    """
    Reassign noise points (label == -1) to the nearest cluster centroid
    (mean of cluster member embeddings).
    """
    labels = labels.copy()
    noise_mask = labels == -1
    n_noise = int(noise_mask.sum())
    if n_noise == 0:
        return labels

    cluster_ids = sorted(set(labels) - {-1})
    if not cluster_ids:
        log.warning("No valid clusters — noise labelled as cluster 0")
        labels[noise_mask] = 0
        return labels

    # Build centroid matrix from mean of each cluster's members
    centroids = np.stack([
        embedding[labels == cid].mean(axis=0) for cid in cluster_ids
    ])  # (K, D)

    noise_pts = embedding[noise_mask]                          # (M, D)
    dists = np.linalg.norm(
        noise_pts[:, None, :] - centroids[None, :, :], axis=2
    )  # (M, K)
    nearest_idx = np.argmin(dists, axis=1)
    labels[noise_mask] = np.array(cluster_ids)[nearest_idx]

    log.info(f"Assigned {n_noise} noise points to nearest cluster centroids")
    return labels


def make_cluster_label(cluster_id: int) -> str:
    if int(cluster_id) == -1:
        return "non_living/noise"
    return f"non_living/cluster_{int(cluster_id):02d}"


# ---------------------------------------------------------------------------
# 8. EcoTaxa ZIP export
# ---------------------------------------------------------------------------

# EcoTaxa columns to include (intersection with what process_pisco_profiles
# saves into the *_crops_metadata.csv)
_ECOTAXA_COLS = [
    "img_file_name",
    "object_id",
    "object_annotation_category",
    "object_annotation_status",
    "object_lat",
    "object_lon",
    "object_date",
    "object_time",
    "object_depth_min",
    "object_depth_max",
    "object_area",
    "object_esd",
    "object_width",
    "object_height",
    "object_major_axis_len",
    "object_minor_axis_len",
    "object_circularity",
    "object_eccentricity",
    "object_solidity",
    "object_mean_intensity",
    "object_prob_1",
    "object_pressure",
    "object_interpolated_t",
    "object_interpolated_s",
    "object_interpolated_o",
    "object_interpolated_chl",
    "sample_cruise",
    "sample_id",
]

# Raw → EcoTaxa column renames (keys = columns in *_crops_metadata.csv)
_RENAME = {
    "full_path":             "object_full_path",
    "filename":              "img_file_name",
    "pressure [dbar]":       "object_pressure",
    "depth [m]":             "object_depth_min",
    "date":                  "object_date",
    "time":                  "object_time",
    "lat":                   "object_lat",
    "lon":                   "object_lon",
    "area":                  "object_area",
    "esd":                   "object_esd",
    "w":                     "object_width",
    "h":                     "object_height",
    "top1":                  "object_annotation_category",
    "prob1":                 "object_prob_1",
    "cruise":                "sample_cruise",
    "interpolated_t":        "object_interpolated_t",
    "interpolated_s":        "object_interpolated_s",
    "interpolated_o":        "object_interpolated_o",
    "interpolated_chl2_raw": "object_interpolated_chl",
}


def _dtype_tag(series: pd.Series) -> str:
    return "[f]" if pd.api.types.is_numeric_dtype(series) else "[t]"


def _insert_dtype_row(df: pd.DataFrame) -> pd.DataFrame:
    """Prepend the EcoTaxa [f]/[t] type-hint row as the first data row."""
    dtype_row = {col: _dtype_tag(df[col]) for col in df.columns}
    return pd.concat(
        [pd.DataFrame([dtype_row]), df],
        ignore_index=True,
    )


def _normalize_object_date(value: object) -> str:
    if pd.isna(value):
        return ""

    if isinstance(value, (int, np.integer)):
        digits = str(int(value))
    elif isinstance(value, (float, np.floating)) and np.isfinite(value):
        digits = str(int(value))
    else:
        s = str(value).strip()
        if not s:
            return ""
        digits = "".join(ch for ch in s if ch.isdigit())

    if len(digits) >= 8:
        return digits[:8]
    return digits.zfill(8) if digits else ""


def _normalize_object_time(value: object) -> str:
    if pd.isna(value):
        return ""

    if isinstance(value, (int, np.integer)):
        digits = str(int(value))
    elif isinstance(value, (float, np.floating)) and np.isfinite(value):
        digits = str(int(value))
    else:
        s = str(value).strip()
        if not s:
            return ""
        digits = "".join(ch for ch in s if ch.isdigit())

    if not digits:
        return ""

    if len(digits) <= 6:
        hhmmss = digits.zfill(6)
        return f"{hhmmss}00"

    if len(digits) == 7:
        return f"0{digits}"

    return digits[:8]


def _build_ecotaxa_df(
    df_sample: pd.DataFrame,
    cluster_labels_map: Dict[str, str],
    path_col: str,
) -> pd.DataFrame:
    """
    Rename raw columns to EcoTaxa names, apply cluster labels for non-living
    crops, and restrict to the EcoTaxa column set.
    """
    df = df_sample.copy()

    # Rename raw column names
    df.rename(columns={k: v for k, v in _RENAME.items() if k in df.columns},
              inplace=True)

    # Ensure minimal required columns
    if "object_annotation_category" not in df.columns:
        df["object_annotation_category"] = "unknown"
    if "object_annotation_status" not in df.columns:
        df["object_annotation_status"] = "predicted"
    if "object_depth_max" not in df.columns and "object_depth_min" in df.columns:
        df["object_depth_max"] = df["object_depth_min"]
    if "sample_id" not in df.columns:
        df["sample_id"] = df.get("sample_cruise", "unknown")

    # Clean annotation category: underscores → spaces
    df["object_annotation_category"] = (
        df["object_annotation_category"]
        .astype(str)
        .str.replace("_", " ", regex=False)
    )

    # Apply cluster labels for non-living crops
    if cluster_labels_map and "object_id" in df.columns:
        cluster_series = df["object_id"].map(cluster_labels_map)
        has_cluster = cluster_series.notna()
        df.loc[has_cluster, "object_annotation_category"] = cluster_series[has_cluster]

    # Flat img_file_name for ZIP (unique basename)
    if "img_file_name" not in df.columns:
        src_col = "object_full_path" if "object_full_path" in df.columns else path_col
        df["img_file_name"] = df[src_col].apply(
            lambda p: os.path.basename(str(p)) if isinstance(p, str) else ""
        )
    else:
        df["img_file_name"] = df["img_file_name"].astype(str).apply(os.path.basename)

    if "object_date" in df.columns:
        df["object_date"] = df["object_date"].apply(_normalize_object_date)
    if "object_time" in df.columns:
        df["object_time"] = df["object_time"].apply(_normalize_object_time)

    # Keep only EcoTaxa columns that are present
    keep = [c for c in _ECOTAXA_COLS if c in df.columns]
    return df[keep]


def _apply_export_filters(
    df_et: pd.DataFrame,
    non_living_classes: Tuple[str, ...] = ("non_living", "detritus"),
    category_filter: str = "all",
    esd_min: Optional[float] = None,
    esd_max: Optional[float] = None,
) -> pd.DataFrame:
    """
    Filter prepared EcoTaxa dataframe by living/non-living and ESD bounds.

    category_filter: "all" | "living" | "non_living"
    """
    if df_et.empty:
        return df_et

    out = df_et.copy()

    if "object_annotation_category" in out.columns:
        nl_mask = non_living_mask(
            out,
            top1_col="object_annotation_category",
            classes=non_living_classes,
        )
    else:
        nl_mask = pd.Series(False, index=out.index)

    mode = str(category_filter or "all").strip().lower()
    if mode == "living":
        out = out[~nl_mask]
    elif mode in {"non_living", "non-living", "nonliving"}:
        out = out[nl_mask]

    if esd_min is not None or esd_max is not None:
        if "object_esd" not in out.columns:
            log.warning(
                "ESD filter requested but object_esd is missing; skipping ESD filtering."
            )
        else:
            esd = pd.to_numeric(out["object_esd"], errors="coerce")
            keep = pd.Series(True, index=out.index)
            if esd_min is not None:
                keep &= esd >= float(esd_min)
            if esd_max is not None:
                keep &= esd <= float(esd_max)
            out = out[keep]

    return out.reset_index(drop=True)


def count_export_particles(
    df_sample: pd.DataFrame,
    cluster_labels_map: Dict[str, str],
    non_living_classes: Tuple[str, ...] = ("non_living", "detritus"),
    category_filter: str = "all",
    esd_min: Optional[float] = None,
    esd_max: Optional[float] = None,
    check_source_exists: bool = False,
) -> Dict[str, int]:
    """Return particle counts after export filtering (before ZIP writing)."""
    path_col = _image_path_col(df_sample)
    df_et = _build_ecotaxa_df(df_sample, cluster_labels_map, path_col)
    if check_source_exists and path_col in df_sample.columns:
        df_et["_src"] = df_sample[path_col].values
        exists_mask = df_et["_src"].apply(lambda p: os.path.isfile(str(p)))
        df_et = df_et[exists_mask].reset_index(drop=True)

    df_filtered = _apply_export_filters(
        df_et=df_et,
        non_living_classes=non_living_classes,
        category_filter=category_filter,
        esd_min=esd_min,
        esd_max=esd_max,
    )

    counts = {"total": int(len(df_filtered)), "living": 0, "non_living": 0}
    if len(df_filtered) == 0:
        return counts

    if "object_annotation_category" in df_filtered.columns:
        nl_mask = non_living_mask(
            df_filtered,
            top1_col="object_annotation_category",
            classes=non_living_classes,
        )
        counts["non_living"] = int(nl_mask.sum())
        counts["living"] = int((~nl_mask).sum())
    else:
        counts["living"] = int(len(df_filtered))
    return counts


def export_ecotaxa_zips(
    df_sample: pd.DataFrame,
    cluster_labels_map: Dict[str, str],
    output_dir: str,
    max_mb: float = 500.0,
    non_living_classes: Tuple[str, ...] = ("non_living", "detritus"),
    category_filter: str = "all",
    esd_min: Optional[float] = None,
    esd_max: Optional[float] = None,
    split_mode: str = "none",
) -> List[str]:
    """
    Write EcoTaxa-compatible ZIP files, split whenever the running image
    size total would exceed *max_mb*.

    Each ZIP contains:
        ecotaxa_validation_part_NN.tsv   — metadata with [f]/[t] type row
        <img_file_name>                  — crop images at ZIP root

    Returns list of created ZIP paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    max_bytes = max_mb * 1024 * 1024

    path_col = _image_path_col(df_sample)
    df_et = _build_ecotaxa_df(df_sample, cluster_labels_map, path_col)
    df_et["_src"] = df_sample[path_col].values

    # Drop rows where source image is missing
    exists_mask = df_et["_src"].apply(lambda p: os.path.isfile(str(p)))
    n_missing = (~exists_mask).sum()
    if n_missing:
        log.warning(f"{n_missing} images not found on disk — excluded from ZIP")
    df_et = df_et[exists_mask].reset_index(drop=True)

    df_et = _apply_export_filters(
        df_et=df_et,
        non_living_classes=non_living_classes,
        category_filter=category_filter,
        esd_min=esd_min,
        esd_max=esd_max,
    )

    if df_et.empty:
        log.warning("Export filter returned zero rows. No ZIPs created.")
        return []

    log.info(
        f"Exporting {len(df_et):,} images  (max {max_mb} MB per ZIP) "
        f"[filter={category_filter}, esd_min={esd_min}, esd_max={esd_max}, split={split_mode}]"
    )

    if "object_annotation_category" in df_et.columns:
        nl_mask_export = non_living_mask(
            df_et,
            top1_col="object_annotation_category",
            classes=non_living_classes,
        )
    else:
        nl_mask_export = pd.Series(False, index=df_et.index)

    split_key = str(split_mode or "none").strip().lower()
    if split_key == "living_non_living":
        groups = [
            ("living", df_et[~nl_mask_export].copy()),
            ("non_living", df_et[nl_mask_export].copy()),
        ]
    else:
        groups = [("all", df_et.copy())]

    zip_paths: List[str] = []

    for group_name, group_df in groups:
        if group_df.empty:
            log.info(f"Skipping empty export group: {group_name}")
            continue

        # Pre-compute file sizes for splitting
        sizes = group_df["_src"].apply(
            lambda p: os.path.getsize(p) if os.path.isfile(p) else 0
        ).values

        # Build (start, end) index ranges for each ZIP part
        row_sets: List[Tuple[int, int]] = []
        chunk_start = 0
        running = 0.0
        for i, sz in enumerate(sizes):
            if running + sz > max_bytes and running > 0:
                row_sets.append((chunk_start, i))
                chunk_start = i
                running = 0.0
            running += sz
        row_sets.append((chunk_start, len(group_df)))

        log.info(f"Group '{group_name}': {len(group_df):,} rows → {len(row_sets)} ZIP file(s)")

        for part_idx, (start, end) in enumerate(row_sets, 1):
            chunk = group_df.iloc[start:end].copy()
            prefix = f"validation_{group_name}" if group_name != "all" else "validation"
            zip_path = os.path.join(output_dir, f"{prefix}_part_{part_idx:02d}.zip")
            tsv_name = f"ecotaxa_{prefix}_part_{part_idx:02d}.tsv"

            # Staging directory
            tmp_dir = os.path.join(output_dir, f"_tmp_{group_name}_{part_idx:02d}")
            os.makedirs(tmp_dir, exist_ok=True)

            # Write TSV with EcoTaxa type-hint row
            meta_cols = [c for c in chunk.columns if not c.startswith("_")]
            df_tsv = _insert_dtype_row(chunk[meta_cols])
            df_tsv.to_csv(os.path.join(tmp_dir, tsv_name), sep="\t", index=False)

            # Copy images into ZIP root directory
            copied = 0
            for _, row in chunk.iterrows():
                src = str(row["_src"])
                dst_name = str(row.get("img_file_name", os.path.basename(src)))
                dst = os.path.join(tmp_dir, dst_name)
                if os.path.isfile(src):
                    try:
                        shutil.copy2(src, dst)
                        copied += 1
                    except Exception as exc:
                        log.warning(f"  Copy failed {src}: {exc}")

            # Create ZIP from staging directory
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for fpath in Path(tmp_dir).rglob("*"):
                    if fpath.is_file():
                        zf.write(fpath, fpath.relative_to(tmp_dir))

            size_mb = os.path.getsize(zip_path) / 1024 / 1024
            log.info(
                f"  {group_name} part {part_idx:02d}: {copied:,} images  →  "
                f"{os.path.basename(zip_path)}  ({size_mb:.1f} MB)"
            )
            shutil.rmtree(tmp_dir, ignore_errors=True)
            zip_paths.append(zip_path)

    return zip_paths


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _metadata_cache_meta_path(cache_path: str) -> str:
    return f"{cache_path}.meta.json"


def _load_metadata_cache(
    cache_path: str,
    expected_meta: Dict[str, object],
) -> Optional[pd.DataFrame]:
    if not os.path.isfile(cache_path):
        return None

    meta_path = _metadata_cache_meta_path(cache_path)
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                cache_meta = json.load(fh)
            mismatches = [
                key for key, value in expected_meta.items()
                if cache_meta.get(key) != value
            ]
            if mismatches:
                log.info(
                    "Metadata cache exists but settings changed "
                    f"({', '.join(mismatches)}). Rebuilding cache."
                )
                return None
        except Exception as exc:
            log.warning(f"Could not read metadata cache sidecar: {exc}. Rebuilding cache.")
            return None
    else:
        log.info("Metadata cache sidecar not found; rebuilding cache for reproducibility.")
        return None

    try:
        df_cached = pd.read_pickle(cache_path)
        log.info(f"Loaded metadata cache: {cache_path}  (rows={len(df_cached):,})")
        return df_cached
    except Exception as exc:
        log.warning(f"Failed to load metadata cache '{cache_path}': {exc}. Rebuilding cache.")
        return None


def _save_metadata_cache(
    df: pd.DataFrame,
    cache_path: str,
    cache_meta: Dict[str, object],
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    df.to_pickle(cache_path)
    with open(_metadata_cache_meta_path(cache_path), "w", encoding="utf-8") as fh:
        json.dump(cache_meta, fh, indent=2, sort_keys=True)
    log.info(f"Metadata cache saved: {cache_path}")


def _auto_resolve_mask_dir(image_paths: List[str]) -> Optional[str]:
    """
    Resolve mask directory as sibling of Deconv_crops/Crops.
    Example: .../_Results/Deconv_crops/<img>.png -> .../_Results/Masks/
    """
    if not image_paths:
        return None

    candidate_counts: Dict[str, int] = {}
    sibling_names = ("Masks", "masks", "Mask", "mask")

    for p in image_paths[:5000]:
        try:
            crop_dir = Path(str(p)).parent
        except Exception:
            continue
        if not crop_dir.name.lower() in {"deconv_crops", "crops"}:
            continue

        base = crop_dir.parent
        for sibling in sibling_names:
            cand = base / sibling
            if cand.is_dir():
                c = str(cand)
                candidate_counts[c] = candidate_counts.get(c, 0) + 1

    if not candidate_counts:
        return None
    return max(candidate_counts.items(), key=lambda kv: kv[1])[0]

def run_pipeline(
    source_root: str,
    output_dir: str,
    metadata_patterns: Optional[Tuple[str, ...]] = None,
    metadata_choice: str = "auto",
    crop_path_token: Optional[str] = "Deconv_crops",
    crop_filter_strict: bool = True,
    rewrite_crops_to_deconv: bool = True,
    metadata_cache_path: Optional[str] = None,
    reuse_metadata_cache: bool = True,
    refresh_metadata_cache: bool = False,
    sample_index_cache_path: Optional[str] = None,
    reuse_sample_index_cache: bool = False,
    refresh_sample_index_cache: bool = False,
    n_samples: int = 100_000,
    seed: int = 42,
    non_living_classes: Tuple[str, ...] = ("non_living", "detritus"),
    ssl_model_path: Optional[str] = None,
    no_train: bool = False,
    ssl_epochs: int = 200,
    ssl_batch_size: int = 256,
    ssl_lr: float = 1e-3,
    ssl_input_size: int = 224,
    ssl_num_workers: int = 4,
    ssl_train_max_images: Optional[int] = 200_000,
    ssl_train_seed: int = 42,
    ssl_aug_profile: str = "default",
    ssl_foreground_crop: bool = False,
    ssl_foreground_threshold: int = 245,
    ssl_foreground_min_pixels: int = 8,
    ssl_mask_dir: Optional[str] = None,
    ssl_mask_suffix: str = "",
    ssl_mask_extension: str = ".png",
    ssl_positive_consistency_weight: float = 0.0,
    ssl_size_aware_canvas: bool = False,
    ssl_canvas_size: int = 512,
    umap_dims: int = 20,
    umap_n_neighbors: int = 30,
    umap_min_dist: float = 0.0,
    min_cluster_size: int = 100,
    hdbscan_min_samples: Optional[int] = None,
    hdbscan_cluster_selection_epsilon: float = 0.0,
    hdbscan_cluster_selection_method: str = "eom",
    size_feature_mode: str = "off",
    size_feature_weight: float = 0.0,
    fallback_kmeans_k: Optional[int] = None,
    fallback_on_all_noise: bool = True,
    keep_noise_as_noise: bool = False,
    max_clusters: Optional[int] = None,
    max_zip_mb: float = 500.0,
    do_export: bool = True,
    stop_requested: Optional[Callable[[], bool]] = None,
    device: Optional[str] = None,
    extract_batch_size: int = 512,
) -> List[str]:
    """
    Run the full validation dataset creation pipeline.
    Returns the list of exported ZIP file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    def _check_stop(stage: str) -> None:
        if stop_requested is not None and stop_requested():
            raise InterruptedError(f"Stopped by user during {stage}.")

    # File handler so everything is also written to a log file
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"create_validation_{ts}.log")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    logging.getLogger().addHandler(fh)

    log.info("=" * 60)
    log.info("PISCO Validation Dataset Creator")
    log.info("=" * 60)
    log.info(f"Source root : {source_root}")
    log.info(f"Output dir  : {output_dir}")
    if metadata_patterns is not None:
        log.info(f"Metadata patterns: {metadata_patterns}")
    log.info(f"Metadata choice: {metadata_choice}")
    log.info(f"Crop path token: {crop_path_token}")
    log.info(f"Rewrite /Crops/ → /Deconv_crops/: {rewrite_crops_to_deconv}")
    if metadata_cache_path is None:
        metadata_cache_path = os.path.join(output_dir, "all_crops_metadata.pkl")
    log.info(f"Metadata cache: {metadata_cache_path}")
    log.info(
        f"Reuse metadata cache: {reuse_metadata_cache} "
        f"(refresh={refresh_metadata_cache})"
    )
    if sample_index_cache_path is None:
        sample_index_cache_path = os.path.join(output_dir, "sample_index.csv")
    log.info(f"Sample index cache: {sample_index_cache_path}")
    log.info(
        f"Reuse sample index cache: {reuse_sample_index_cache} "
        f"(refresh={refresh_sample_index_cache})"
    )
    log.info(f"Samples     : {n_samples:,}")
    log.info(f"Non-living  : {non_living_classes}")
    log.info(f"Log file    : {log_path}")
    _check_stop("setup")

    # ── Step 1: Collect ──────────────────────────────────────────────────
    log.info("\n── Step 1: Load/collect metadata ─────────────────────────")
    cache_meta = {
        "source_root": os.path.abspath(source_root),
        "metadata_patterns": list(metadata_patterns) if metadata_patterns is not None else None,
        "metadata_choice": metadata_choice,
        "crop_path_token": crop_path_token,
        "crop_filter_strict": bool(crop_filter_strict),
        "rewrite_crops_to_deconv": bool(rewrite_crops_to_deconv),
    }

    df_all: Optional[pd.DataFrame] = None
    if reuse_metadata_cache and not refresh_metadata_cache:
        df_all = _load_metadata_cache(metadata_cache_path, expected_meta=cache_meta)

    if df_all is None:
        df_all = collect_all_crops(
            source_root=source_root,
            metadata_patterns=metadata_patterns,
            metadata_choice=metadata_choice,
            crop_path_token=crop_path_token,
            crop_filter_strict=crop_filter_strict,
            rewrite_crops_to_deconv=rewrite_crops_to_deconv,
        )
        if reuse_metadata_cache:
            try:
                _save_metadata_cache(df_all, metadata_cache_path, cache_meta=cache_meta)
            except Exception as exc:
                log.warning(f"Could not save metadata cache: {exc}")
    top1 = _top1_col(df_all)
    nl_all = non_living_mask(df_all, top1, non_living_classes)

    # Auto-augment class tokens if metadata uses "not living" style labels.
    if int(nl_all.sum()) == 0:
        probe_values = (
            df_all[top1]
            .astype(str)
            .str.lower()
            .str.replace(r"[_\-]+", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        if probe_values.str.contains(r"\bnot\s+living\b", na=False).any():
            augmented = tuple(list(non_living_classes) + ["not living"])
            log.info(
                "Detected label variant 'not living' in metadata; "
                "adding it to non-living classes for this run"
            )
            nl_all = non_living_mask(df_all, top1, augmented)
            non_living_classes = augmented

    log.info(
        f"Non-living in full collection: {nl_all.sum():,} / {len(df_all):,} "
        f"({100 * nl_all.mean():.1f}%)"
    )
    _check_stop("metadata collection")

    # ── Step 2: Sample ───────────────────────────────────────────────────
    log.info("\n── Step 2: Random sample ─────────────────────────────────")
    sample_index_path = os.path.join(output_dir, "sample_index.csv")
    loaded_sample_cache = False
    df_sample: pd.DataFrame

    if reuse_sample_index_cache and not refresh_sample_index_cache and os.path.isfile(sample_index_cache_path):
        try:
            df_sample = pd.read_csv(sample_index_cache_path, low_memory=False)
            loaded_sample_cache = True
            log.info(
                f"Loaded sample index cache: {sample_index_cache_path} "
                f"(rows={len(df_sample):,})"
            )
            if len(df_sample) != n_samples:
                log.warning(
                    f"Cached sample size ({len(df_sample):,}) differs from --n-samples ({n_samples:,}). "
                    "Using cached sample as requested."
                )
        except Exception as exc:
            log.warning(f"Could not load sample index cache: {exc}. Rebuilding sample.")
            loaded_sample_cache = False

    if not loaded_sample_cache:
        df_sample = random_sample(df_all, n=n_samples, seed=seed)
        if reuse_sample_index_cache:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(sample_index_cache_path)), exist_ok=True)
                df_sample.to_csv(sample_index_cache_path, index=False)
                log.info(f"Sample index cache saved → {sample_index_cache_path}")
            except Exception as exc:
                log.warning(f"Could not save sample index cache: {exc}")

    top1_sample = _top1_col(df_sample)
    nl_sample = non_living_mask(df_sample, top1_sample, non_living_classes)
    log.info(
        f"Non-living in sample: {nl_sample.sum():,} / {len(df_sample):,} "
        f"({100 * nl_sample.mean():.1f}%)"
    )
    if sample_index_cache_path != sample_index_path or not os.path.isfile(sample_index_path):
        df_sample.to_csv(sample_index_path, index=False)
    log.info(f"Sample index → {sample_index_path}")
    _check_stop("sampling")

    cluster_labels_map: Dict[str, str] = {}
    assignments_path: Optional[str] = None
    h5_path = os.path.join(output_dir, "features_nonliving.h5")

    # ── Step 3/4/5: SSL + features + clustering (only when non-living exists) ──
    path_col = _image_path_col(df_all)
    nl_count_all = int(nl_all.sum())
    nl_count_sample = int(nl_sample.sum())

    if nl_count_all == 0 or nl_count_sample == 0:
        log.warning(
            "No non-living crops detected (full set or sampled set). "
            "Skipping SSL training and clustering; export will keep original labels."
        )
    else:
        log.info("\n── Step 3: SimCLR training on non-living crops ───────────")
        if ssl_model_path is None:
            ssl_model_path = os.path.join(output_dir, f"ssl_nonliving_{ts}.pth")

        use_imagenet_fallback = False
        if no_train:
            if not os.path.isfile(ssl_model_path):
                abs_model = os.path.abspath(ssl_model_path)
                log.warning(
                    f"--no-train set but ssl-model not found: {ssl_model_path} "
                    f"(absolute: {abs_model}). Falling back to ImageNet ResNet-50 features."
                )
                use_imagenet_fallback = True
            else:
                log.info(f"Skipping training — using existing model: {ssl_model_path}")
        else:
            _check_stop("SSL training preparation")
            nl_paths_all = df_all.loc[nl_all, path_col].dropna().tolist()

            if ssl_train_max_images is not None and len(nl_paths_all) > ssl_train_max_images:
                rng = np.random.default_rng(ssl_train_seed)
                idx = rng.choice(len(nl_paths_all), size=ssl_train_max_images, replace=False)
                nl_paths_all = [nl_paths_all[i] for i in idx]
                log.info(
                    f"Using random subset for SSL training: {len(nl_paths_all):,} images "
                    f"(seed={ssl_train_seed})"
                )
            else:
                log.info(f"Using all non-living images for SSL training: {len(nl_paths_all):,}")

            exists_mask_all = [os.path.isfile(str(p)) for p in nl_paths_all]
            n_exist_all = int(sum(exists_mask_all))
            if n_exist_all < len(nl_paths_all):
                log.warning(
                    f"SSL training paths missing on disk: {len(nl_paths_all) - n_exist_all:,} "
                    f"(keeping {n_exist_all:,})"
                )
            nl_paths_all = [p for p, ok in zip(nl_paths_all, exists_mask_all) if ok]
            if not nl_paths_all:
                raise RuntimeError(
                    "No existing non-living training images after path rewrite/filter. "
                    "Check Deconv_crops availability or disable rewrite/filter options."
                )

            if os.path.isfile(ssl_model_path):
                log.info(f"Checkpoint already exists, skipping training: {ssl_model_path}")
            else:
                resolved_mask_dir = ssl_mask_dir
                if resolved_mask_dir is None and ssl_foreground_crop:
                    resolved_mask_dir = _auto_resolve_mask_dir(nl_paths_all)
                    if resolved_mask_dir is not None:
                        log.info(f"Auto-resolved SSL mask dir: {resolved_mask_dir}")
                    else:
                        log.info(
                            "SSL mask dir auto-resolution found no sibling Masks dir; "
                            "falling back to threshold-based foreground mask."
                        )

                log.info(f"Training on {len(nl_paths_all):,} non-living images ...")
                train_ssl_model(
                    image_paths=nl_paths_all,
                    output_path=ssl_model_path,
                    pretrained_backbone=True,
                    epochs=ssl_epochs,
                    batch_size=ssl_batch_size,
                    lr=ssl_lr,
                    input_size=ssl_input_size,
                    num_workers=ssl_num_workers,
                    device=device,
                    augmentation_profile=ssl_aug_profile,
                    foreground_crop=ssl_foreground_crop,
                    foreground_threshold=ssl_foreground_threshold,
                    foreground_min_pixels=ssl_foreground_min_pixels,
                    mask_dir=resolved_mask_dir,
                    mask_suffix=ssl_mask_suffix,
                    mask_extension=ssl_mask_extension,
                    positive_consistency_weight=ssl_positive_consistency_weight,
                    size_aware_canvas=ssl_size_aware_canvas,
                    canvas_size=ssl_canvas_size,
                    stop_requested=stop_requested,
                )

        _check_stop("post-SSL")

        log.info("\n── Step 4: Feature extraction ────────────────────────────")
        nl_df_sample = df_sample[nl_sample].copy()
        nl_paths_sample = nl_df_sample[path_col].dropna().tolist()
        nl_oids_sample = (
            nl_df_sample["object_id"].astype(str).tolist()
            if "object_id" in nl_df_sample.columns
            else [str(i) for i in nl_df_sample.index]
        )

        exists_mask_sample = [os.path.isfile(str(p)) for p in nl_paths_sample]
        n_exist_sample = int(sum(exists_mask_sample))
        if n_exist_sample < len(nl_paths_sample):
            log.warning(
                f"Sample non-living paths missing on disk: {len(nl_paths_sample) - n_exist_sample:,} "
                f"(keeping {n_exist_sample:,})"
            )
        nl_paths_sample = [p for p, ok in zip(nl_paths_sample, exists_mask_sample) if ok]
        nl_oids_sample = [o for o, ok in zip(nl_oids_sample, exists_mask_sample) if ok]
        if not nl_paths_sample:
            raise RuntimeError(
                "No existing non-living sample images after path rewrite/filter. "
                "Check Deconv_crops availability or disable rewrite/filter options."
            )

        if os.path.isfile(h5_path):
            log.info(f"Loading cached features: {h5_path}")
            features, feat_paths, feat_oids = load_features_h5(h5_path)
        else:
            if use_imagenet_fallback:
                features, feat_paths = extract_features_imagenet(
                    image_paths=nl_paths_sample,
                    batch_size=extract_batch_size,
                    num_workers=ssl_num_workers,
                    device=device,
                    input_size=ssl_input_size,
                    size_aware_canvas=ssl_size_aware_canvas,
                    canvas_size=ssl_canvas_size,
                    stop_requested=stop_requested,
                )
            else:
                features, feat_paths = extract_features(
                    image_paths=nl_paths_sample,
                    checkpoint_path=ssl_model_path,
                    batch_size=extract_batch_size,
                    num_workers=ssl_num_workers,
                    device=device,
                    input_size=ssl_input_size,
                    size_aware_canvas=ssl_size_aware_canvas,
                    canvas_size=ssl_canvas_size,
                    stop_requested=stop_requested,
                )
            path_to_oid = dict(zip(nl_paths_sample, nl_oids_sample))
            feat_oids = [path_to_oid.get(p, str(i)) for i, p in enumerate(feat_paths)]
            save_features_h5(features, feat_paths, feat_oids, h5_path)

        _check_stop("feature extraction")

        features_for_clustering = augment_features_with_object_area(
            features=features,
            feat_oids=feat_oids,
            df_nonliving_sample=nl_df_sample,
            mode=size_feature_mode,
            weight=size_feature_weight,
        )

        log.info("\n── Step 5: UMAP + HDBSCAN clustering ─────────────────────")
        embedding, raw_labels, hdb_model, all_noise = cluster_features(
            features_for_clustering,
            n_umap_components=umap_dims,
            umap_n_neighbors=umap_n_neighbors,
            umap_min_dist=umap_min_dist,
            min_cluster_size=min_cluster_size,
            hdbscan_min_samples=hdbscan_min_samples,
            hdbscan_cluster_selection_epsilon=hdbscan_cluster_selection_epsilon,
            hdbscan_cluster_selection_method=hdbscan_cluster_selection_method,
            max_clusters=max_clusters,
            random_state=seed,
        )
        _check_stop("clustering")

        if keep_noise_as_noise:
            final_labels = raw_labels.copy()
            n_noise_keep = int(np.sum(final_labels == -1))
            log.info(f"Keeping noise as dedicated label (-1): {n_noise_keep:,} points")
        else:
            final_labels = assign_noise_to_nearest(embedding, raw_labels, hdb_model)

        if (
            all_noise
            and fallback_on_all_noise
            and fallback_kmeans_k is not None
            and not keep_noise_as_noise
        ):
            log.warning(
                "HDBSCAN returned all-noise labels. Applying KMeans fallback "
                f"with k={fallback_kmeans_k}."
            )
            final_labels = kmeans_fallback_labels(
                embedding=embedding,
                k=fallback_kmeans_k,
                random_state=seed,
            )

        # Ensure assignment paths consistently point to deconvolved crops,
        # including when features are loaded from older cached HDF5 files.
        feat_paths_rewritten = [
            re.sub(r"([/\\])Crops([/\\])", r"\1Deconv_crops\2", str(p))
            for p in feat_paths
        ]
        n_paths_rewritten = int(
            sum(p0 != p1 for p0, p1 in zip(feat_paths, feat_paths_rewritten))
        )
        feat_paths = feat_paths_rewritten
        if n_paths_rewritten:
            log.info(
                f"Cluster assignment paths rewritten to Deconv_crops: {n_paths_rewritten:,}"
            )

        cluster_labels_map = {
            oid: make_cluster_label(int(lbl))
            for oid, lbl in zip(feat_oids, final_labels)
        }

        unique_ids, counts = np.unique(final_labels, return_counts=True)
        log.info(f"{'Cluster label':<30s} {'Count':>8s}")
        log.info("-" * 40)
        for uid, cnt in sorted(zip(unique_ids, counts), key=lambda x: -x[1]):
            log.info(f"  {make_cluster_label(int(uid)):<28s} {cnt:>8,}")

        assignments_path = os.path.join(output_dir, "cluster_assignments.csv")
        pd.DataFrame({
            "object_id":     feat_oids,
            "image_path":    feat_paths,
            "cluster_raw":   raw_labels,
            "cluster_final": final_labels,
            "cluster_label": [cluster_labels_map[oid] for oid in feat_oids],
        }).to_csv(assignments_path, index=False)
        log.info(f"Cluster assignments → {assignments_path}")

        try:
            log.info("Computing 2-D UMAP for visualisation ...")
            reducer_2d = umap.UMAP(
                n_components=2, n_neighbors=30, min_dist=0.1,
                metric="cosine", random_state=0,
            )
            emb_2d = reducer_2d.fit_transform(features_for_clustering)
            viz_path = os.path.join(output_dir, "umap_2d.csv")
            pd.DataFrame({
                "object_id":     feat_oids,
                "umap_x":        emb_2d[:, 0],
                "umap_y":        emb_2d[:, 1],
                "cluster_raw":   raw_labels,
                "cluster_final": final_labels,
                "cluster_label": [cluster_labels_map[oid] for oid in feat_oids],
            }).to_csv(viz_path, index=False)
            log.info(f"2-D UMAP → {viz_path}")

            # Remove stale explorer from older runs; HTML viewer generation is disabled.
            explorer_path = os.path.join(output_dir, "embedding_explorer.html")
            if os.path.isfile(explorer_path):
                try:
                    os.remove(explorer_path)
                    log.info(f"Removed legacy embedding explorer HTML: {explorer_path}")
                except Exception as exc:
                    log.warning(f"Could not remove legacy explorer HTML: {exc}")
        except Exception as exc:
            log.warning(f"2-D UMAP visualisation failed: {exc}")

    # ── Step 6: Export EcoTaxa ZIPs ──────────────────────────────────────
    zip_paths: List[str] = []
    if do_export:
        _check_stop("export")
        log.info("\n── Step 6: EcoTaxa ZIP export ────────────────────────────")
        zip_dir = os.path.join(output_dir, "ecotaxa_zips")
        zip_paths = export_ecotaxa_zips(
            df_sample=df_sample,
            cluster_labels_map=cluster_labels_map,
            output_dir=zip_dir,
            max_mb=max_zip_mb,
            non_living_classes=non_living_classes,
        )
    else:
        log.info("\n── Step 6: EcoTaxa ZIP export (skipped: do_export=False) ──")

    # ── Summary ──────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("DONE")
    log.info("=" * 60)
    log.info(f"  Sample index        : {sample_index_path}")
    if os.path.exists(h5_path):
        log.info(f"  Features HDF5       : {h5_path}")
    if assignments_path is not None:
        log.info(f"  Cluster assignments : {assignments_path}")
    log.info(f"  EcoTaxa ZIPs ({len(zip_paths)}):")
    for zp in zip_paths:
        mb = os.path.getsize(zp) / 1024 / 1024
        log.info(f"    {zp}  ({mb:.1f} MB)")
    log.info(f"  Log                 : {log_path}")

    logging.getLogger().removeHandler(fh)
    return zip_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Create PISCO validation dataset with non-living clustering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
# Full run (train SimCLR + cluster + export):
  python create_validation_dataset.py \\
      --source /mnt/filer \\
      --output /mnt/filer/validation_dataset

# Reuse existing SSL model:
  python create_validation_dataset.py \\
      --source /mnt/filer \\
      --output /mnt/filer/validation_dataset \\
      --ssl-model models/ssl_nonliving.pth

# Dry run (1 000 samples, no training):
  python create_validation_dataset.py \\
      --source /mnt/filer \\
      --output /tmp/dry_run \\
      --n-samples 1000 \\
      --no-train \\
      --ssl-model models/ssl_nonliving.pth
""",
    )

    # I/O
    p.add_argument("--source", required=True,
                   help="Root directory to scan for *_crops_metadata.csv files")
    p.add_argument("--output", required=True,
                   help="Output directory (ZIPs, features.h5, logs)")
    p.add_argument(
        "--metadata-patterns",
        nargs="+",
        default=["*_crops_metadata.csv", "*_metadata.csv", "ecotaxa_metadata.tsv", "*_ecotaxa.tsv"],
        help="Filename patterns searched recursively under --source for metadata files",
    )
    p.add_argument(
        "--metadata-choice",
        choices=["auto", "crops", "ecotaxa"],
        default="auto",
        help="When both files exist in an EcoTaxa folder: pick crop metadata, EcoTaxa TSV, or auto",
    )
    p.add_argument(
        "--crop-path-token",
        default="Deconv_crops",
        help="Path filter token (default: Deconv_crops); use 'off' to disable",
    )
    p.add_argument(
        "--no-crop-path-filter",
        action="store_true",
        help="Disable crop-path token filtering",
    )
    p.add_argument(
        "--crop-filter-strict",
        dest="crop_filter_strict",
        action="store_true",
        default=True,
        help="Fail if --crop-path-token has zero matches (default: enabled)",
    )
    p.add_argument(
        "--no-crop-filter-strict",
        dest="crop_filter_strict",
        action="store_false",
        help="Do not fail when crop-path token has zero matches",
    )
    p.add_argument(
        "--no-rewrite-crops-to-deconv",
        action="store_true",
        help="Do not rewrite /Crops/ paths to /Deconv_crops/ before filtering",
    )
    p.add_argument(
        "--metadata-cache-path",
        default=None,
        help="Path to cache full metadata dataframe (default: <output>/all_crops_metadata.pkl)",
    )
    p.add_argument(
        "--no-reuse-metadata-cache",
        action="store_true",
        help="Disable metadata cache loading/saving",
    )
    p.add_argument(
        "--refresh-metadata-cache",
        action="store_true",
        help="Rebuild metadata cache even if one already exists",
    )
    p.add_argument(
        "--sample-index-cache-path",
        default=None,
        help="Path to cached sample index CSV (default: <output>/sample_index.csv)",
    )
    p.add_argument(
        "--reuse-sample-index-cache",
        action="store_true",
        help="Load/save sample index cache to skip random sampling on later runs",
    )
    p.add_argument(
        "--refresh-sample-index-cache",
        action="store_true",
        help="Rebuild sample index cache even if one already exists",
    )

    # Sampling
    p.add_argument("--n-samples", type=int, default=100_000,
                   help="Random subset size (default: 100 000)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")

    # Non-living detection
    p.add_argument("--non-living-classes", nargs="+",
                   default=["non_living", "detritus"],
                   help="ViT class substrings that identify non-living particles")

    # SSL model
    p.add_argument("--ssl-model", default=None,
                   help="Existing SimCLR checkpoint path. If omitted, a new one is trained.")
    p.add_argument("--no-train", action="store_true",
                   help="Skip SSL training (requires --ssl-model)")
    p.add_argument("--ssl-epochs", type=int, default=200)
    p.add_argument("--ssl-batch-size", type=int, default=256)
    p.add_argument("--ssl-lr", type=float, default=1e-3)
    p.add_argument("--ssl-input-size", type=int, default=224)
    p.add_argument("--ssl-workers", type=int, default=4)
    p.add_argument(
        "--ssl-train-max-images",
        type=int,
        default=200000,
        help="Maximum non-living images used to train SimCLR (set 0 or negative for all)",
    )
    p.add_argument(
        "--ssl-train-seed",
        type=int,
        default=42,
        help="Random seed for selecting SSL training subset",
    )
    p.add_argument(
        "--ssl-aug-profile",
        choices=["default", "sparse_grayscale"],
        default="default",
        help="SSL augmentation profile (default or sparse_grayscale for sparse dark-on-white crops)",
    )
    p.add_argument(
        "--ssl-foreground-crop",
        action="store_true",
        help="Crop each training image around foreground before augmentations",
    )
    p.add_argument(
        "--ssl-foreground-threshold",
        type=int,
        default=245,
        help="Foreground threshold on grayscale intensity for auto-mask (< threshold)",
    )
    p.add_argument(
        "--ssl-foreground-min-pixels",
        type=int,
        default=8,
        help="Minimum foreground pixels required to apply foreground crop",
    )
    p.add_argument(
        "--ssl-mask-dir",
        default=None,
        help="Optional directory of external masks for foreground cropping (default: auto-resolve sibling 'Masks' next to Deconv_crops)",
    )
    p.add_argument(
        "--ssl-mask-suffix",
        default="",
        help="Optional suffix inserted before mask extension (e.g. '_mask')",
    )
    p.add_argument(
        "--ssl-mask-extension",
        default=".png",
        help="File extension for external masks (default: .png)",
    )
    p.add_argument(
        "--ssl-positive-consistency-weight",
        type=float,
        default=0.0,
        help="Extra positive-pair cosine consistency term weight added to NT-Xent loss",
    )
    p.add_argument(
        "--ssl-size-aware-canvas",
        action="store_true",
        help="Place crops on a fixed canvas before augmentations/inference to preserve relative pixel size",
    )
    p.add_argument(
        "--ssl-canvas-size",
        type=int,
        default=512,
        help="Canvas side length used when --ssl-size-aware-canvas is enabled",
    )

    # Clustering
    p.add_argument("--umap-dims", type=int, default=64,
                   help="UMAP output dimensionality (default: 64)")
    p.add_argument("--umap-n-neighbors", type=int, default=30,
                   help="UMAP n_neighbors (default: 30)")
    p.add_argument("--umap-min-dist", type=float, default=0.0,
                   help="UMAP min_dist (default: 0.0)")
    p.add_argument("--min-cluster-size", type=int, default=100,
                   help="HDBSCAN min_cluster_size (default: 100)")
    p.add_argument("--hdbscan-min-samples", type=int, default=None,
                   help="HDBSCAN min_samples (default: None = equals min_cluster_size behavior)")
    p.add_argument("--hdbscan-cluster-selection-epsilon", type=float, default=0.0,
                   help="HDBSCAN cluster_selection_epsilon (default: 0.0)")
    p.add_argument("--hdbscan-cluster-selection-method", choices=["eom", "leaf"], default="eom",
                   help="HDBSCAN cluster_selection_method (default: eom)")
    p.add_argument(
        "--size-feature-mode",
        choices=["off", "append_log_area"],
        default="off",
        help="Optional size-aware channel for clustering (default: off)",
    )
    p.add_argument(
        "--size-feature-weight",
        type=float,
        default=0.0,
        help="Weight of normalized log(object_area) when size-feature-mode is append_log_area",
    )
    p.add_argument("--fallback-kmeans-k", type=int, default=None,
                   help="If HDBSCAN returns all-noise, assign KMeans labels with this K")
    p.add_argument("--no-fallback-on-all-noise", action="store_true",
                   help="Disable automatic KMeans fallback when HDBSCAN has zero clusters")
    p.add_argument("--keep-noise-as-noise", action="store_true",
                   help="Keep HDBSCAN noise points as -1 (non_living/noise) instead of assigning nearest cluster")
    p.add_argument("--max-clusters", type=int, default=None,
                   help="Maximum number of clusters. Closest clusters are merged "
                        "iteratively until this limit is reached. No limit if omitted.")

    # Export
    p.add_argument("--max-zip-mb", type=float, default=500.0,
                   help="Maximum size per EcoTaxa ZIP in MB (default: 500)")

    # Hardware
    p.add_argument("--device", default=None,
                   help="Compute device: cuda | cpu (auto-detected if omitted)")
    p.add_argument("--extract-batch-size", type=int, default=512,
                   help="Batch size for feature extraction inference")

    args = p.parse_args()

    ssl_train_max_images = args.ssl_train_max_images
    if ssl_train_max_images is not None and ssl_train_max_images <= 0:
        ssl_train_max_images = None

    run_pipeline(
        source_root=args.source,
        output_dir=args.output,
        metadata_patterns=tuple(args.metadata_patterns),
        metadata_choice=args.metadata_choice,
        crop_path_token=None if args.no_crop_path_filter else args.crop_path_token,
        crop_filter_strict=args.crop_filter_strict,
        rewrite_crops_to_deconv=not args.no_rewrite_crops_to_deconv,
        metadata_cache_path=args.metadata_cache_path,
        reuse_metadata_cache=not args.no_reuse_metadata_cache,
        refresh_metadata_cache=args.refresh_metadata_cache,
        sample_index_cache_path=args.sample_index_cache_path,
        reuse_sample_index_cache=args.reuse_sample_index_cache,
        refresh_sample_index_cache=args.refresh_sample_index_cache,
        n_samples=args.n_samples,
        seed=args.seed,
        non_living_classes=tuple(args.non_living_classes),
        ssl_model_path=args.ssl_model,
        no_train=args.no_train,
        ssl_epochs=args.ssl_epochs,
        ssl_batch_size=args.ssl_batch_size,
        ssl_lr=args.ssl_lr,
        ssl_input_size=args.ssl_input_size,
        ssl_num_workers=args.ssl_workers,
        ssl_train_max_images=ssl_train_max_images,
        ssl_train_seed=args.ssl_train_seed,
        ssl_aug_profile=args.ssl_aug_profile,
        ssl_foreground_crop=args.ssl_foreground_crop,
        ssl_foreground_threshold=args.ssl_foreground_threshold,
        ssl_foreground_min_pixels=args.ssl_foreground_min_pixels,
        ssl_mask_dir=args.ssl_mask_dir,
        ssl_mask_suffix=args.ssl_mask_suffix,
        ssl_mask_extension=args.ssl_mask_extension,
        ssl_positive_consistency_weight=args.ssl_positive_consistency_weight,
        ssl_size_aware_canvas=args.ssl_size_aware_canvas,
        ssl_canvas_size=args.ssl_canvas_size,
        umap_dims=args.umap_dims,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        min_cluster_size=args.min_cluster_size,
        hdbscan_min_samples=args.hdbscan_min_samples,
        hdbscan_cluster_selection_epsilon=args.hdbscan_cluster_selection_epsilon,
        hdbscan_cluster_selection_method=args.hdbscan_cluster_selection_method,
        size_feature_mode=args.size_feature_mode,
        size_feature_weight=args.size_feature_weight,
        fallback_kmeans_k=args.fallback_kmeans_k,
        fallback_on_all_noise=not args.no_fallback_on_all_noise,
        keep_noise_as_noise=args.keep_noise_as_noise,
        max_clusters=args.max_clusters,
        max_zip_mb=args.max_zip_mb,
        device=args.device,
        extract_batch_size=args.extract_batch_size,
    )


if __name__ == "__main__":
    main()
