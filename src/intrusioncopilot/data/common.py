"""Shared, memory-safe preprocessing helpers for IntrusionCopilot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from intrusioncopilot.config import settings


def load_dataset_config(dataset_key: str) -> dict[str, Any]:
    """Load dataset-specific preprocessing config from configs/datasets.json."""
    cfg_path = settings.resolve_path(settings.configs_dir) / "datasets.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing dataset config file: {cfg_path}")
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg = dict(data[dataset_key])
    cfg["random_seed"] = int(data.get("random_seed", settings.random_seed))
    return cfg


def clip_scaled_values(X: np.ndarray, low: float = -10.0, high: float = 10.0) -> np.ndarray:
    """Top-level, joblib-serializable clipper to prevent extreme outliers from dominating AE MSE."""
    arr = np.asarray(X, dtype=np.float32)
    return np.clip(arr, low, high).astype(np.float32, copy=False)


def downcast_numeric_columns(df: pd.DataFrame, exclude: set[str] | None = None) -> pd.DataFrame:
    """Downcast float64->float32 and safe integer columns->int32 without silent overflow."""
    skip = exclude or set()
    i32_min, i32_max = np.iinfo(np.int32).min, np.iinfo(np.int32).max

    for col in df.columns:
        if col in skip:
            continue
        dtype = df[col].dtype
        if pd.api.types.is_float_dtype(dtype):
            df[col] = df[col].astype(np.float32)
        elif pd.api.types.is_integer_dtype(dtype):
            cmin, cmax = int(df[col].min()), int(df[col].max())
            if i32_min <= cmin and cmax <= i32_max:
                df[col] = df[col].astype(np.int32)
    return df


def hash_feature_rows(df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    """Compute deterministic 64-bit hashes over feature columns for deduplication & leakage checks."""
    return pd.util.hash_pandas_object(df[feature_cols], index=False)


def deduplicate_and_resolve_conflicts(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove exact duplicates and ambiguous rows where identical features map to conflicting labels."""
    initial_rows = len(df)
    work = df.copy()
    work["_row_hash"] = hash_feature_rows(work, feature_cols).to_numpy()

    # 1. Remove exact (feature + label) duplicates
    work = work.drop_duplicates(subset=["_row_hash", label_col], keep="first")
    exact_dups_removed = initial_rows - len(work)

    # 2. Identify and drop ambiguous feature vectors carrying conflicting labels
    label_nunique = work.groupby("_row_hash")[label_col].nunique()
    conflicting_hashes = set(label_nunique[label_nunique > 1].index.tolist())
    if conflicting_hashes:
        conflict_mask = work["_row_hash"].isin(conflicting_hashes)
        conflicting_rows_removed = int(conflict_mask.sum())
        work = work.loc[~conflict_mask]
    else:
        conflicting_rows_removed = 0

    work = work.drop(columns=["_row_hash"]).reset_index(drop=True)
    stats = {
        "initial_rows": initial_rows,
        "exact_duplicates_removed": int(exact_dups_removed),
        "conflicting_label_hashes": len(conflicting_hashes),
        "conflicting_label_rows_removed": conflicting_rows_removed,
        "remaining_unique_rows": int(len(work)),
    }
    return work, stats


def save_report(report: dict[str, Any], markdown_text: str, stem: str) -> tuple[Path, Path]:
    """Write preprocessing validation report in both JSON and Markdown under reports/preprocessing/."""
    out_dir = settings.resolve_path(settings.reports_dir) / "preprocessing"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(markdown_text, encoding="utf-8")
    return json_path, md_path