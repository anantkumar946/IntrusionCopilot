"""Memory-safe, leakage-aware preprocessing pipeline for CICIDS2017.

Run via:
    uv run python -m intrusioncopilot.data.preprocess_cicids
"""

from __future__ import annotations

import gc
import json
import re
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler

from intrusioncopilot.config import settings
from intrusioncopilot.data.common import (
    clip_scaled_values,
    deduplicate_and_resolve_conflicts,
    hash_feature_rows,
    load_dataset_config,
    save_report,
)
from intrusioncopilot.data.inspect import read_csv_with_encoding
from intrusioncopilot.utils.logging import get_logger

logger = get_logger(__name__)

KNOWN_ZERO_VARIANCE_COLS = [
    "Bwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "Fwd Avg Bulk Rate",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
]
DUPLICATE_HEADER_COL = "Fwd Header Length.1"
ENGINEERED_FEATURES = ["fwd_bwd_pkt_ratio", "fwd_bwd_byte_ratio", "total_packets"]


def normalize_cicids_label(label: str) -> str:
    """Clean corrupted non-ASCII en-dash characters and extra spaces in CICIDS2017 labels."""
    text = str(label).strip()
    text = re.sub(r"\s*[\ufffd\x96\u2013\u2014]+\s*", " - ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def add_cicids_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add row-wise traffic asymmetry features (zero cross-row leakage)."""
    fwd_pkts = df["Total Fwd Packets"].astype(np.float64)
    bwd_pkts = df["Total Backward Packets"].astype(np.float64)
    fwd_bytes = df["Total Length of Fwd Packets"].astype(np.float64)
    bwd_bytes = df["Total Length of Bwd Packets"].astype(np.float64)

    df["fwd_bwd_pkt_ratio"] = ((fwd_pkts + 1.0) / (bwd_pkts + 1.0)).astype(np.float32)
    df["fwd_bwd_byte_ratio"] = ((fwd_bytes + 1.0) / (bwd_bytes + 1.0)).astype(np.float32)
    df["total_packets"] = (fwd_pkts + bwd_pkts).astype(np.float32)
    return df


def build_cicids_preprocessors(cfg: dict[str, Any]) -> tuple[Pipeline, Pipeline]:
    """Build numeric preprocessors for supervised trees and tabular Autoencoder."""
    low, high = cfg["ae_clip_bounds"]

    tree_preprocessor = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
        ]
    )

    ae_preprocessor = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
            (
                "clip",
                FunctionTransformer(
                    clip_scaled_values,
                    kw_args={"low": float(low), "high": float(high)},
                    validate=False,
                    feature_names_out="one-to-one",
                ),
            ),
        ]
    )
    return tree_preprocessor, ae_preprocessor


def _class_dist(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts().items()}


def _render_markdown(report: dict[str, Any]) -> str:
    splits = report["splits"]
    leakage = report["leakage_audit"]
    dedup = report["deduplication"]
    lines = [
        "# CICIDS2017 Preprocessing Validation Report\n",
        f"- **Random Seed:** `{report['random_seed']}`",
        f"- **Dropped Columns ({len(report['cleaning']['dropped_columns'])}):** `{report['cleaning']['dropped_columns']}`",
        f"- **Sentinel Negative Columns Preserved (`-1`):** `{report['cleaning']['sentinel_negative_columns_preserved']}`",
        f"- **Inf Values Replaced with NaN (train-imputed):** `{report['cleaning']['inf_values_replaced_with_nan']:,}`",
        f"- **Invalid Negative Values Replaced with NaN (train-imputed):** `{report['cleaning']['invalid_negative_values_replaced_with_nan']:,}`",
        f"- **Engineered Features ({len(report['features']['engineered_features'])}):** `{report['features']['engineered_features']}`",
        f"- **Total Model Input Features:** `{report['features']['total_input_features']}` "
        f"(Tree dim: `{report['preprocessors']['tree_output_dim']}` | AE dim: `{report['preprocessors']['ae_output_dim']}`)\n",
        "## Row Reduction & Deduplication Summary\n",
        "| Stage | Row Count |",
        "|---|---:|",
        f"| Raw rows across {report['raw_files_processed']} CSVs | {dedup['raw_total_rows']:,} |",
        f"| In-file exact duplicates removed during streaming | {dedup['in_file_duplicates_removed']:,} |",
        f"| Cross-file exact duplicates removed | {dedup['global_exact_duplicates_removed']:,} |",
        f"| Conflicting-label rows removed | {dedup['conflicting_label_rows_removed']:,} |",
        f"| Final clean unique rows | {dedup['final_clean_rows']:,} |\n",
        "## Final Split Sizes & Leakage Audit\n",
        "| Split | Rows | Benign (0) | Malicious (1) | Parquet Size (MB) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("train", "val", "test"):
        s = splits[name]
        b = s["binary_distribution"]
        lines.append(
            f"| `{name}` | {s['rows']:,} | {b.get('0', 0):,} | {b.get('1', 0):,} | {s['parquet_size_mb']} |"
        )
    lines.extend(
        [
            "",
            f"- **Zero-Leakage Verification:** train∩val = `{leakage['train_val_feature_overlap']}`, "
            f"train∩test = `{leakage['train_test_feature_overlap']}`, "
            f"val∩test = `{leakage['val_test_feature_overlap']}`",
            f"- **Tree preprocessor fit strictly on:** `train` ({splits['train']['rows']:,} rows)",
            f"- **Autoencoder preprocessor fit strictly on:** `train` Benign rows only ({splits['train']['binary_distribution'].get('0', 0):,} rows)\n",
            "## Multiclass Distribution (`attack_cat`)\n",
            "| Class | ID | Train | Val | Test |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for cls_name, cls_id in report["label_mapping"]["attack_cat_to_id"].items():
        lines.append(
            f"| `{cls_name}` | {cls_id} | "
            f"{splits['train']['multiclass_distribution'].get(cls_name, 0):,} | "
            f"{splits['val']['multiclass_distribution'].get(cls_name, 0):,} | "
            f"{splits['test']['multiclass_distribution'].get(cls_name, 0):,} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_cicids_preprocessing() -> dict[str, Any]:
    """Execute the memory-safe, leakage-aware CICIDS2017 preprocessing pipeline."""
    cfg = load_dataset_config("cicids2017")
    seed = int(cfg["random_seed"])
    raw_dir = settings.resolve_path(settings.cicids_raw_dir)
    processed_dir = settings.resolve_path(settings.processed_dir) / "cicids2017"
    artifact_dir = settings.resolve_path(settings.artifacts_dir) / "preprocessors"
    processed_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    target_col = cfg["target_column"]
    benign_name = cfg["benign_class_name"]
    sentinel_neg_cols = set(cfg["sentinel_negative_columns"])

    chunks: list[pd.DataFrame] = []
    raw_total_rows = 0
    in_file_dups_removed = 0
    total_inf_replaced = 0
    total_invalid_neg_replaced = 0
    dup_header_identical_all = True
    raw_feature_cols: list[str] | None = None

    logger.info("Streaming %d CICIDS2017 CSVs for memory-safe cleaning...", len(csv_files))
    for path in csv_files:
        df, _ = read_csv_with_encoding(path)
        df.columns = [c.strip() for c in df.columns]
        raw_total_rows += len(df)

        # Verify duplicate header equality before dropping
        if DUPLICATE_HEADER_COL in df.columns and "Fwd Header Length" in df.columns:
            if not df[DUPLICATE_HEADER_COL].equals(df["Fwd Header Length"]):
                dup_header_identical_all = False
            df = df.drop(columns=[DUPLICATE_HEADER_COL])

        drop_now = [c for c in KNOWN_ZERO_VARIANCE_COLS if c in df.columns]
        df = df.drop(columns=drop_now)

        # Normalize label strings to clean ASCII
        df[target_col] = df[target_col].map(normalize_cicids_label)

        feat_cols = [c for c in df.columns if c != target_col]
        if raw_feature_cols is None:
            raw_feature_cols = feat_cols
        elif feat_cols != raw_feature_cols:
            raise ValueError(f"Schema mismatch in {path.name}")

        # Clean +/-inf and invalid negatives column-by-column, downcasting to float32 immediately
        for col in feat_cols:
            s = pd.to_numeric(df[col], errors="coerce")
            arr = s.to_numpy(dtype=np.float64, copy=True)

            inf_mask = np.isinf(arr)
            n_inf = int(inf_mask.sum())
            if n_inf > 0:
                total_inf_replaced += n_inf
                arr[inf_mask] = np.nan

            if col not in sentinel_neg_cols:
                neg_mask = arr < 0
                n_neg = int(neg_mask.sum())
                if n_neg > 0:
                    total_invalid_neg_replaced += n_neg
                    arr[neg_mask] = np.nan

            df[col] = arr.astype(np.float32)

        # Per-file exact deduplication to reduce peak RAM before global concat
        before_dedup = len(df)
        df = df.drop_duplicates(keep="first").reset_index(drop=True)
        in_file_dups_removed += before_dedup - len(df)
        chunks.append(df)
        logger.info("  Cleaned %-58s -> %8d unique rows", path.name, len(df))

    assert raw_feature_cols is not None
    logger.info("Concatenating cleaned float32 chunks...")
    combined = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    # Global cross-file deduplication & conflicting-label resolution
    logger.info("Running global deduplication and conflict resolution on %d rows...", len(combined))
    clean_df, global_dedup_stats = deduplicate_and_resolve_conflicts(combined, raw_feature_cols, target_col)
    del combined
    gc.collect()

    # Deterministic label mapping (BENIGN = 0, attack classes sorted alphabetically 1..K)
    attack_classes = sorted(c for c in clean_df[target_col].unique() if c != benign_name)
    attack_cat_to_id = {benign_name: 0, **{cls: idx for idx, cls in enumerate(attack_classes, start=1)}}
    id_to_attack_cat = {str(v): k for k, v in attack_cat_to_id.items()}

    # Add row-wise engineered features
    clean_df = add_cicids_engineered_features(clean_df)
    all_feature_cols = raw_feature_cols + ENGINEERED_FEATURES

    clean_df["y_binary"] = (clean_df[target_col] != benign_name).astype(np.int8)
    clean_df["y_multiclass"] = clean_df[target_col].map(attack_cat_to_id).astype(np.int8)
    clean_df["attack_cat"] = clean_df[target_col].astype(str)
    clean_df = clean_df.drop(columns=[target_col])

    # Stratified 70 / 15 / 15 Train / Val / Test split
    val_ratio = float(cfg["val_ratio"])
    test_ratio = float(cfg["test_ratio"])
    holdout_ratio = val_ratio + test_ratio

    train_df, temp_df = train_test_split(
        clean_df,
        test_size=holdout_ratio,
        random_state=seed,
        stratify=clean_df["y_multiclass"],
    )
    del clean_df
    gc.collect()

    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_ratio / holdout_ratio,
        random_state=seed,
        stratify=temp_df["y_multiclass"],
    )
    del temp_df
    gc.collect()

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Check if any additional column became zero-variance on train split
    extra_zero_var = [
        c for c in raw_feature_cols if train_df[c].nunique(dropna=True) <= 1
    ]
    if extra_zero_var:
        logger.warning("Dropping train-split zero-variance columns: %s", extra_zero_var)
        raw_feature_cols = [c for c in raw_feature_cols if c not in extra_zero_var]
        all_feature_cols = raw_feature_cols + ENGINEERED_FEATURES
        for frame in (train_df, val_df, test_df):
            frame.drop(columns=extra_zero_var, inplace=True)

    # Zero-leakage audit across final splits
    logger.info("Verifying zero feature-hash overlap across Train / Val / Test...")
    h_train = set(hash_feature_rows(train_df, raw_feature_cols).tolist())
    h_val = set(hash_feature_rows(val_df, raw_feature_cols).tolist())
    h_test = set(hash_feature_rows(test_df, raw_feature_cols).tolist())
    leakage_audit = {
        "train_val_feature_overlap": len(h_train & h_val),
        "train_test_feature_overlap": len(h_train & h_test),
        "val_test_feature_overlap": len(h_val & h_test),
    }
    del h_train, h_val, h_test
    gc.collect()
    if any(v != 0 for v in leakage_audit.values()):
        raise RuntimeError(f"Data leakage detected across CICIDS2017 splits: {leakage_audit}")

    # Fit preprocessors strictly on training split (AE on Benign train rows only)
    logger.info("Fitting tree & Autoencoder preprocessors on training split...")
    tree_prep, ae_prep = build_cicids_preprocessors(cfg)

    X_train_arr = train_df[all_feature_cols].to_numpy(dtype=np.float32, copy=False)
    tree_prep.fit(X_train_arr)
    del X_train_arr
    gc.collect()

    benign_mask = (train_df["y_binary"] == 0).to_numpy()
    n_train_benign = int(benign_mask.sum())
    X_train_benign = train_df.loc[benign_mask, all_feature_cols].to_numpy(dtype=np.float32, copy=False)
    ae_prep.fit(X_train_benign)
    del X_train_benign
    gc.collect()

    # Validate finite transforms on val and test (memory-safe batch check)
    for split_name, frame in (("val", val_df), ("test", test_df)):
        arr = frame[all_feature_cols].to_numpy(dtype=np.float32, copy=False)
        t_out = tree_prep.transform(arr)
        a_out = ae_prep.transform(arr)
        if not np.isfinite(t_out).all() or not np.isfinite(a_out).all():
            raise ValueError(f"Non-finite values detected after transforming CICIDS2017 {split_name} split!")
        del arr, t_out, a_out
        gc.collect()

    # Save processed splits in Parquet
    save_cols = all_feature_cols + ["y_binary", "y_multiclass", "attack_cat"]
    split_meta: dict[str, Any] = {}
    for split_name, frame in (("train", train_df), ("val", val_df), ("test", test_df)):
        pq_path = processed_dir / f"{split_name}.parquet"
        frame[save_cols].to_parquet(pq_path, index=False, engine="pyarrow")
        split_meta[split_name] = {
            "rows": int(len(frame)),
            "parquet_path": str(pq_path.relative_to(settings.project_root)),
            "parquet_size_mb": round(pq_path.stat().st_size / (1024 * 1024), 2),
            "binary_distribution": _class_dist(frame["y_binary"]),
            "multiclass_distribution": _class_dist(frame["attack_cat"]),
        }
        logger.info("  Saved %-5s split: %9d rows (%.2f MB)", split_name, len(frame), split_meta[split_name]["parquet_size_mb"])

    # Save preprocessor bundle & label map
    bundle_path = artifact_dir / "cicids2017_prep.joblib"
    bundle = {
        "dataset": "cicids2017",
        "random_seed": seed,
        "feature_columns": all_feature_cols,
        "raw_feature_columns": raw_feature_cols,
        "numeric_columns": all_feature_cols,
        "categorical_columns": [],
        "engineered_columns": ENGINEERED_FEATURES,
        "tree_feature_names": all_feature_cols,
        "ae_feature_names": all_feature_cols,
        "attack_cat_to_id": attack_cat_to_id,
        "id_to_attack_cat": id_to_attack_cat,
        "tree_preprocessor": tree_prep,
        "ae_preprocessor": ae_prep,
    }
    joblib.dump(bundle, bundle_path)

    label_map_path = artifact_dir / "cicids2017_labels.json"
    label_map_path.write_text(
        json.dumps({"attack_cat_to_id": attack_cat_to_id, "id_to_attack_cat": id_to_attack_cat}, indent=2),
        encoding="utf-8",
    )

    dropped_cols = KNOWN_ZERO_VARIANCE_COLS + [DUPLICATE_HEADER_COL] + extra_zero_var
    report: dict[str, Any] = {
        "dataset": "CICIDS2017",
        "random_seed": seed,
        "raw_files_processed": len(csv_files),
        "config": cfg,
        "cleaning": {
            "dropped_columns": dropped_cols,
            "duplicate_header_verified_identical": dup_header_identical_all,
            "destination_port_retained_rationale": "Retained as protocol/service signal and alert network_context; IPs/FlowIDs/timestamps are absent in MachineLearningCVE.",
            "sentinel_negative_columns_preserved": sorted(sentinel_neg_cols),
            "inf_values_replaced_with_nan": total_inf_replaced,
            "invalid_negative_values_replaced_with_nan": total_invalid_neg_replaced,
            "non_ascii_web_attack_labels_normalized": True,
        },
        "deduplication": {
            "raw_total_rows": raw_total_rows,
            "in_file_duplicates_removed": in_file_dups_removed,
            "global_exact_duplicates_removed": global_dedup_stats["exact_duplicates_removed"],
            "conflicting_label_hashes": global_dedup_stats["conflicting_label_hashes"],
            "conflicting_label_rows_removed": global_dedup_stats["conflicting_label_rows_removed"],
            "final_clean_rows": global_dedup_stats["remaining_unique_rows"],
        },
        "features": {
            "total_input_features": len(all_feature_cols),
            "raw_retained_features": raw_feature_cols,
            "numeric_features": all_feature_cols,
            "categorical_features": [],
            "engineered_features": ENGINEERED_FEATURES,
        },
        "preprocessors": {
            "bundle_path": str(bundle_path.relative_to(settings.project_root)),
            "tree_output_dim": len(all_feature_cols),
            "ae_output_dim": len(all_feature_cols),
            "tree_fit_rows": split_meta["train"]["rows"],
            "ae_fit_rows_benign_only": n_train_benign,
        },
        "label_mapping": {
            "attack_cat_to_id": attack_cat_to_id,
            "id_to_attack_cat": id_to_attack_cat,
        },
        "leakage_audit": leakage_audit,
        "splits": split_meta,
    }

    md_text = _render_markdown(report)
    json_path, md_path = save_report(report, md_text, "cicids2017_report")
    logger.info("CICIDS2017 preprocessing complete. Report saved to %s and %s", json_path, md_path)
    print("\n" + md_text)
    return report


if __name__ == "__main__":
    run_cicids_preprocessing()