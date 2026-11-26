"""Leakage-aware preprocessing pipeline for UNSW-NB15.

Run via:
    uv run python -m intrusioncopilot.data.preprocess_unsw
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, RobustScaler

from intrusioncopilot.config import settings
from intrusioncopilot.data.common import (
    clip_scaled_values,
    deduplicate_and_resolve_conflicts,
    downcast_numeric_columns,
    hash_feature_rows,
    load_dataset_config,
    save_report,
)
from intrusioncopilot.utils.logging import get_logger

logger = get_logger(__name__)

ENGINEERED_FEATURES = ["bytes_ratio", "pkts_ratio", "total_bytes", "total_pkts"]


def clean_unsw_frame(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Standardize columns, drop ID leakage column, normalize categoricals, and verify targets."""
    work = df.copy()
    work.columns = [c.strip() for c in work.columns]

    dropped = [c for c in cfg["drop_columns"] if c in work.columns]
    work = work.drop(columns=dropped)

    # Normalize categorical strings
    work["proto"] = work["proto"].astype(str).str.strip().str.lower()
    work["service"] = (
        work["service"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"-": "unknown", "": "unknown", "nan": "unknown"})
    )
    work["state"] = work["state"].astype(str).str.strip()

    # Standardize target columns
    mc_col = cfg["multiclass_target"]
    bin_col = cfg["binary_target"]
    benign_name = cfg["benign_class_name"]

    work[mc_col] = work[mc_col].astype(str).str.strip()
    work[bin_col] = work[bin_col].astype(np.int8)

    # Audit alignment between binary label and attack_cat
    expected_bin = (work[mc_col] != benign_name).astype(np.int8)
    mismatch_count = int((work[bin_col] != expected_bin).sum())
    if mismatch_count > 0:
        logger.warning("Found %d binary/multiclass label mismatches; aligning binary label to %s.", mismatch_count, mc_col)
        work[bin_col] = expected_bin

    # Replace any unexpected +/-inf with NaN on numeric columns
    cat_set = set(cfg["categorical_columns"]) | {mc_col, bin_col}
    num_cols = [c for c in work.columns if c not in cat_set]
    inf_replaced = 0
    for col in num_cols:
        arr = work[col].to_numpy(dtype=np.float64, copy=False)
        n_inf = int(np.isinf(arr).sum())
        if n_inf > 0:
            inf_replaced += n_inf
            work[col] = work[col].replace([np.inf, -np.inf], np.nan)

    work = downcast_numeric_columns(work, exclude=cat_set)
    return work, {
        "dropped_columns": dropped,
        "label_mismatch_count": mismatch_count,
        "inf_values_replaced_with_nan": inf_replaced,
    }


def add_unsw_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-grounded, row-wise traffic asymmetry features (zero cross-row leakage)."""
    work = df.copy()
    sbytes = work["sbytes"].astype(np.float64)
    dbytes = work["dbytes"].astype(np.float64)
    spkts = work["spkts"].astype(np.float64)
    dpkts = work["dpkts"].astype(np.float64)

    work["bytes_ratio"] = ((sbytes + 1.0) / (dbytes + 1.0)).astype(np.float32)
    work["pkts_ratio"] = ((spkts + 1.0) / (dpkts + 1.0)).astype(np.float32)
    work["total_bytes"] = (sbytes + dbytes).astype(np.float32)
    work["total_pkts"] = (spkts + dpkts).astype(np.float32)
    return work


def build_unsw_preprocessors(
    numeric_cols: list[str],
    categorical_cols: list[str],
    cfg: dict[str, Any],
) -> tuple[ColumnTransformer, ColumnTransformer]:
    """Build separate tree-model and Autoencoder preprocessors sharing the exact input schema."""
    low, high = cfg["ae_clip_bounds"]

    tree_preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
                        (
                            "encode",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                                dtype=np.float32,
                            ),
                        ),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    ae_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
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
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
                        (
                            "ohe",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=float(cfg["ae_ohe_min_frequency"]),
                                sparse_output=False,
                                dtype=np.float32,
                            ),
                        ),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return tree_preprocessor, ae_preprocessor


def _class_dist(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts().items()}


def _render_markdown(report: dict[str, Any]) -> str:
    splits = report["splits"]
    leakage = report["leakage_audit"]
    lines = [
        "# UNSW-NB15 Preprocessing Validation Report\n",
        f"- **Random Seed:** `{report['random_seed']}`",
        f"- **Dropped Columns:** `{report['cleaning']['dropped_columns']}`",
        f"- **Engineered Features ({len(report['features']['engineered_features'])}):** `{report['features']['engineered_features']}`",
        f"- **Total Model Input Features:** `{report['features']['total_input_features']}` "
        f"(Tree matrix: `{report['preprocessors']['tree_output_dim']}` dims | "
        f"Autoencoder matrix: `{report['preprocessors']['ae_output_dim']}` dims)\n",
        "## Row Reduction & Deduplication Summary\n",
        "| Stage | Official Train CSV | Official Test CSV |",
        "|---|---:|---:|",
        f"| Raw rows | {report['deduplication']['train_file']['initial_rows']:,} | {report['deduplication']['test_file']['initial_rows']:,} |",
        f"| Exact duplicates removed | {report['deduplication']['train_file']['exact_duplicates_removed']:,} | {report['deduplication']['test_file']['exact_duplicates_removed']:,} |",
        f"| Conflicting-label rows removed | {report['deduplication']['train_file']['conflicting_label_rows_removed']:,} | {report['deduplication']['test_file']['conflicting_label_rows_removed']:,} |",
        f"| Cross-split train overlap removed from test | 0 | {report['deduplication']['cross_split_overlap_removed_from_test']:,} |",
        f"| Final clean rows | {splits['train']['rows'] + splits['val']['rows']:,} | {splits['test']['rows']:,} |\n",
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


def run_unsw_preprocessing() -> dict[str, Any]:
    """Execute the full leakage-aware UNSW-NB15 preprocessing pipeline."""
    cfg = load_dataset_config("unsw_nb15")
    seed = int(cfg["random_seed"])
    raw_dir = settings.resolve_path(settings.unsw_raw_dir)
    processed_dir = settings.resolve_path(settings.processed_dir) / "unsw_nb15"
    artifact_dir = settings.resolve_path(settings.artifacts_dir) / "preprocessors"
    processed_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    train_csv = raw_dir / cfg["train_filename"]
    test_csv = raw_dir / cfg["test_filename"]
    logger.info("Loading UNSW-NB15 raw files: %s & %s", train_csv.name, test_csv.name)

    raw_train = pd.read_csv(train_csv, encoding="utf-8-sig", low_memory=False)
    raw_test = pd.read_csv(test_csv, encoding="utf-8-sig", low_memory=False)

    clean_train, train_clean_stats = clean_unsw_frame(raw_train, cfg)
    clean_test, test_clean_stats = clean_unsw_frame(raw_test, cfg)

    mc_col = cfg["multiclass_target"]
    bin_col = cfg["binary_target"]
    benign_name = cfg["benign_class_name"]
    raw_feature_cols = [c for c in clean_train.columns if c not in {mc_col, bin_col}]

    # 1. Deduplicate and drop conflicting-label feature rows independently
    dedup_train, train_dedup_stats = deduplicate_and_resolve_conflicts(clean_train, raw_feature_cols, mc_col)
    dedup_test, test_dedup_stats = deduplicate_and_resolve_conflicts(clean_test, raw_feature_cols, mc_col)

    # 2. Remove exact feature-vector overlap from held-out test set
    train_hashes = set(hash_feature_rows(dedup_train, raw_feature_cols).tolist())
    test_hashes = hash_feature_rows(dedup_test, raw_feature_cols)
    overlap_mask = test_hashes.isin(train_hashes)
    cross_overlap_removed = int(overlap_mask.sum())
    dedup_test = dedup_test.loc[~overlap_mask].reset_index(drop=True)
    logger.info("Removed %d cross-split overlapping feature rows from UNSW-NB15 test set.", cross_overlap_removed)

    # 3. Deterministic label mapping (Benign/Normal = 0, attacks sorted alphabetically 1..K)
    attack_classes = sorted(c for c in dedup_train[mc_col].unique() if c != benign_name)
    attack_cat_to_id = {benign_name: 0, **{cls: idx for idx, cls in enumerate(attack_classes, start=1)}}
    id_to_attack_cat = {str(v): k for k, v in attack_cat_to_id.items()}

    # 4. Row-wise engineered features
    dedup_train = add_unsw_engineered_features(dedup_train)
    dedup_test = add_unsw_engineered_features(dedup_test)

    categorical_cols = list(cfg["categorical_columns"])
    all_feature_cols = raw_feature_cols + ENGINEERED_FEATURES
    numeric_cols = [c for c in all_feature_cols if c not in categorical_cols]

    # Attach standardized target columns
    for frame in (dedup_train, dedup_test):
        frame["y_binary"] = frame[bin_col].astype(np.int8)
        frame["y_multiclass"] = frame[mc_col].map(attack_cat_to_id).astype(np.int8)
        frame["attack_cat"] = frame[mc_col].astype(str)

    # 5. Stratified Train / Val split from deduplicated training file
    train_df, val_df = train_test_split(
        dedup_train,
        test_size=float(cfg["val_ratio_from_train"]),
        random_state=seed,
        stratify=dedup_train["y_multiclass"],
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = dedup_test.reset_index(drop=True)

    # 6. Zero-leakage audit across final splits
    h_train = set(hash_feature_rows(train_df, raw_feature_cols).tolist())
    h_val = set(hash_feature_rows(val_df, raw_feature_cols).tolist())
    h_test = set(hash_feature_rows(test_df, raw_feature_cols).tolist())
    leakage_audit = {
        "train_val_feature_overlap": len(h_train & h_val),
        "train_test_feature_overlap": len(h_train & h_test),
        "val_test_feature_overlap": len(h_val & h_test),
    }
    if any(v != 0 for v in leakage_audit.values()):
        raise RuntimeError(f"Data leakage detected across UNSW-NB15 splits: {leakage_audit}")

    # 7. Fit preprocessors strictly on training split (AE on benign training rows only)
    tree_prep, ae_prep = build_unsw_preprocessors(numeric_cols, categorical_cols, cfg)
    X_train = train_df[all_feature_cols]
    X_train_benign = train_df.loc[train_df["y_binary"] == 0, all_feature_cols]

    X_train_tree = tree_prep.fit_transform(X_train).astype(np.float32)
    X_train_ae = ae_prep.fit_transform(X_train_benign).astype(np.float32)

    # Sanity-check transform on val & test (verifies unseen states like ACC/CLO do not fail)
    X_val_tree = tree_prep.transform(val_df[all_feature_cols]).astype(np.float32)
    X_test_tree = tree_prep.transform(test_df[all_feature_cols]).astype(np.float32)
    X_test_ae = ae_prep.transform(test_df[all_feature_cols]).astype(np.float32)

    for name, arr in [
        ("X_train_tree", X_train_tree),
        ("X_val_tree", X_val_tree),
        ("X_test_tree", X_test_tree),
        ("X_train_ae", X_train_ae),
        ("X_test_ae", X_test_ae),
    ]:
        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite values detected in transformed matrix {name}!")

    # 8. Save processed splits in Parquet (features + standardized targets)
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

    # 9. Persist preprocessor bundle & label map
    bundle_path = artifact_dir / "unsw_nb15_prep.joblib"
    bundle = {
        "dataset": "unsw_nb15",
        "random_seed": seed,
        "feature_columns": all_feature_cols,
        "raw_feature_columns": raw_feature_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "engineered_columns": ENGINEERED_FEATURES,
        "tree_feature_names": list(tree_prep.get_feature_names_out()),
        "ae_feature_names": list(ae_prep.get_feature_names_out()),
        "attack_cat_to_id": attack_cat_to_id,
        "id_to_attack_cat": id_to_attack_cat,
        "tree_preprocessor": tree_prep,
        "ae_preprocessor": ae_prep,
    }
    joblib.dump(bundle, bundle_path)

    label_map_path = artifact_dir / "unsw_nb15_labels.json"
    label_map_path.write_text(
        json.dumps({"attack_cat_to_id": attack_cat_to_id, "id_to_attack_cat": id_to_attack_cat}, indent=2),
        encoding="utf-8",
    )

    report: dict[str, Any] = {
        "dataset": "UNSW-NB15",
        "random_seed": seed,
        "config": cfg,
        "cleaning": {
            "dropped_columns": train_clean_stats["dropped_columns"],
            "train_label_mismatches": train_clean_stats["label_mismatch_count"],
            "test_label_mismatches": test_clean_stats["label_mismatch_count"],
            "inf_values_replaced": train_clean_stats["inf_values_replaced_with_nan"]
            + test_clean_stats["inf_values_replaced_with_nan"],
            "service_dash_mapped_to": "unknown",
        },
        "deduplication": {
            "train_file": train_dedup_stats,
            "test_file": test_dedup_stats,
            "cross_split_overlap_removed_from_test": cross_overlap_removed,
        },
        "features": {
            "total_input_features": len(all_feature_cols),
            "raw_retained_features": raw_feature_cols,
            "numeric_features": numeric_cols,
            "categorical_features": categorical_cols,
            "engineered_features": ENGINEERED_FEATURES,
        },
        "preprocessors": {
            "bundle_path": str(bundle_path.relative_to(settings.project_root)),
            "tree_output_dim": int(X_train_tree.shape[1]),
            "ae_output_dim": int(X_train_ae.shape[1]),
            "tree_fit_rows": int(len(X_train)),
            "ae_fit_rows_benign_only": int(len(X_train_benign)),
        },
        "label_mapping": {
            "attack_cat_to_id": attack_cat_to_id,
            "id_to_attack_cat": id_to_attack_cat,
        },
        "leakage_audit": leakage_audit,
        "splits": split_meta,
    }

    md_text = _render_markdown(report)
    json_path, md_path = save_report(report, md_text, "unsw_nb15_report")
    logger.info("UNSW-NB15 preprocessing complete. Report saved to %s and %s", json_path, md_path)
    print("\n" + md_text)
    return report


if __name__ == "__main__":
    run_unsw_preprocessing()