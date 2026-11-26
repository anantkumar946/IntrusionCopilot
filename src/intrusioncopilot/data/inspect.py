"""Read-only, memory-safe dataset inspector for UNSW-NB15 and CICIDS2017.

Run via:
    uv run python -m intrusioncopilot.data.inspect
"""

from __future__ import annotations

import gc
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from intrusioncopilot.config import settings
from intrusioncopilot.utils.logging import get_logger

logger = get_logger(__name__)

ENCODINGS_TO_TRY = ("utf-8-sig", "utf-8", "cp1252", "latin1")


def read_csv_with_encoding(path: Path) -> tuple[pd.DataFrame, str]:
    """Read a CSV trying common encodings without mutating the source file."""
    last_error: Exception | None = None
    for enc in ENCODINGS_TO_TRY:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            return df, enc
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"Failed to decode {path.name} with {ENCODINGS_TO_TRY}: {last_error}")


def _audit_numeric_anomalies(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Return counts of NaN, +inf/-inf, and negative values per column."""
    nan_counts: dict[str, int] = {}
    inf_counts: dict[str, int] = {}
    neg_counts: dict[str, int] = {}

    for col in df.columns:
        series = df[col]
        n_nan = int(series.isna().sum())
        if n_nan > 0:
            nan_counts[col] = n_nan

        if pd.api.types.is_numeric_dtype(series):
            arr = series.to_numpy(dtype=np.float64, copy=False)
            n_inf = int(np.isinf(arr).sum())
            n_neg = int((arr < 0).sum())
            if n_inf > 0:
                inf_counts[col] = n_inf
            if n_neg > 0:
                neg_counts[col] = n_neg
        else:
            # Detect string-encoded infinities in object columns (common in CICIDS2017)
            lowered = series.dropna().astype(str).str.strip().str.lower()
            n_str_inf = int(lowered.isin({"inf", "-inf", "infinity", "-infinity"}).sum())
            if n_str_inf > 0:
                inf_counts[col] = n_str_inf

    return {"nan": nan_counts, "inf": inf_counts, "negative": neg_counts}


def inspect_unsw_nb15(raw_dir: Path) -> dict[str, Any]:
    """Inspect official UNSW-NB15 CSV files independently and audit cross-file overlaps."""
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    logger.info("Inspecting UNSW-NB15 (%d files) in %s", len(csv_files), raw_dir)
    file_reports: dict[str, Any] = {}
    loaded: dict[str, pd.DataFrame] = {}

    for path in csv_files:
        df, enc = read_csv_with_encoding(path)
        loaded[path.name] = df

        cols_with_ws = [c for c in df.columns if c != c.strip()]
        anomalies = _audit_numeric_anomalies(df)
        feature_cols_no_id = [c for c in df.columns if c.lower() != "id"]
        cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        zero_var_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]

        cat_summary = {
            c: {
                "nunique": int(df[c].nunique(dropna=False)),
                "top_values": {str(k): int(v) for k, v in df[c].value_counts(dropna=False).head(12).items()},
                "has_dash_placeholder": bool((df[c].astype(str).str.strip() == "-").any()),
            }
            for c in cat_cols
        }

        dup_full = int(df.duplicated().sum())
        dup_no_id = int(df.duplicated(subset=feature_cols_no_id).sum())

        file_reports[path.name] = {
            "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
            "encoding": enc,
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "column_names": list(df.columns),
            "columns_with_whitespace": cols_with_ws,
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "numeric_columns_count": int(df.shape[1] - len(cat_cols)),
            "categorical_columns": cat_cols,
            "categorical_summary": cat_summary,
            "zero_variance_columns": zero_var_cols,
            "anomalies": anomalies,
            "exact_duplicates_all_cols": dup_full,
            "exact_duplicates_excluding_id": dup_no_id,
            "binary_label_distribution": (
                {str(k): int(v) for k, v in df["label"].value_counts(dropna=False).items()}
                if "label" in df.columns
                else {}
            ),
            "attack_cat_distribution": (
                {repr(str(k)): int(v) for k, v in df["attack_cat"].value_counts(dropna=False).items()}
                if "attack_cat" in df.columns
                else {}
            ),
        }

    # Cross-file schema, unseen category, and overlap checks
    cross_file: dict[str, Any] = {}
    names = list(loaded.keys())
    if len(names) == 2:
        df_a, df_b = loaded[names[0]], loaded[names[1]]
        cross_file["identical_columns"] = list(df_a.columns) == list(df_b.columns)

        common_no_id = [c for c in df_a.columns if c in df_b.columns and c.lower() != "id"]
        hash_a = set(pd.util.hash_pandas_object(df_a[common_no_id], index=False).tolist())
        hash_b = set(pd.util.hash_pandas_object(df_b[common_no_id], index=False).tolist())
        overlap_hashes = hash_a.intersection(hash_b)

        a_in_b = int(pd.util.hash_pandas_object(df_a[common_no_id], index=False).isin(overlap_hashes).sum())
        b_in_a = int(pd.util.hash_pandas_object(df_b[common_no_id], index=False).isin(overlap_hashes).sum())
        cross_file["cross_file_overlap_excluding_id"] = {
            "files": names,
            "unique_shared_row_hashes": len(overlap_hashes),
            f"rows_in_{names[0]}_also_in_{names[1]}": a_in_b,
            f"rows_in_{names[1]}_also_in_{names[0]}": b_in_a,
        }

        unseen_cats: dict[str, Any] = {}
        for col in [c for c in common_no_id if not pd.api.types.is_numeric_dtype(df_a[c])]:
            vals_a = set(df_a[col].dropna().astype(str).unique())
            vals_b = set(df_b[col].dropna().astype(str).unique())
            unseen_cats[col] = {
                f"only_in_{names[0]}": sorted(vals_a - vals_b)[:20],
                f"only_in_{names[1]}": sorted(vals_b - vals_a)[:20],
            }
        cross_file["categorical_value_differences"] = unseen_cats

    del loaded
    gc.collect()

    return {"dataset": "UNSW-NB15", "raw_dir": str(raw_dir), "files": file_reports, "cross_file_audit": cross_file}


def inspect_cicids2017(raw_dir: Path) -> dict[str, Any]:
    """Stream CICIDS2017 CSVs one by one for memory-safe schema, anomaly, and duplicate inspection."""
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    logger.info("Inspecting CICIDS2017 (%d files) in %s (streaming file-by-file)", len(csv_files), raw_dir)

    file_reports: dict[str, Any] = {}
    reference_cols: list[str] | None = None
    schema_mismatches: list[str] = []

    total_rows = 0
    total_nan: Counter[str] = Counter()
    total_inf: Counter[str] = Counter()
    total_neg: Counter[str] = Counter()
    object_cols_seen: set[str] = set()
    col_min: dict[str, float] = {}
    col_max: dict[str, float] = {}

    label_counts_raw: Counter[str] = Counter()
    label_counts_dedup: Counter[str] = Counter()
    seen_hashes: set[int] = set()
    total_cross_file_dups = 0

    for path in csv_files:
        logger.info("  Reading %s (%.2f MB)...", path.name, path.stat().st_size / (1024 * 1024))
        df, enc = read_csv_with_encoding(path)
        raw_cols = list(df.columns)
        stripped_cols = [c.strip() for c in raw_cols]
        ws_cols = [c for c in raw_cols if c != c.strip()]

        if reference_cols is None:
            reference_cols = stripped_cols
        elif stripped_cols != reference_cols:
            schema_mismatches.append(path.name)

        df.columns = stripped_cols
        anomalies = _audit_numeric_anomalies(df)

        for col, cnt in anomalies["nan"].items():
            total_nan[col] += cnt
        for col, cnt in anomalies["inf"].items():
            total_inf[col] += cnt
        for col, cnt in anomalies["negative"].items():
            total_neg[col] += cnt

        non_num = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        object_cols_seen.update(non_num)

        # Track global min/max on finite numeric values to find zero-variance columns across all 8 files
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                s = df[col].replace([np.inf, -np.inf], np.nan).dropna()
                if not s.empty:
                    cmin, cmax = float(s.min()), float(s.max())
                    col_min[col] = min(col_min.get(col, cmin), cmin)
                    col_max[col] = max(col_max.get(col, cmax), cmax)

        label_col = "Label" if "Label" in df.columns else df.columns[-1]
        raw_labels = df[label_col].astype(str)
        file_label_dist = {repr(k): int(v) for k, v in raw_labels.value_counts(dropna=False).items()}
        for k, v in raw_labels.value_counts(dropna=False).items():
            label_counts_raw[repr(k)] += int(v)

        # Memory-safe exact duplicate audit across all 8 files via 64-bit row hashes
        hashes = pd.util.hash_pandas_object(df, index=False).to_numpy()
        in_file_dups = int(len(hashes) - len(np.unique(hashes)))

        # Vectorized check against global seen_hashes
        label_list = raw_labels.tolist()
        file_new_unique = 0
        for h, lbl in zip(hashes.tolist(), label_list):
            if h not in seen_hashes:
                seen_hashes.add(h)
                label_counts_dedup[repr(lbl)] += 1
                file_new_unique += 1
            else:
                total_cross_file_dups += 1

        total_rows += len(df)
        file_reports[path.name] = {
            "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
            "encoding": enc,
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "whitespace_columns_count": len(ws_cols),
            "non_numeric_columns": non_num,
            "in_file_exact_duplicates": in_file_dups,
            "new_unique_rows_added": file_new_unique,
            "nan_columns": anomalies["nan"],
            "inf_columns": anomalies["inf"],
            "label_distribution": file_label_dist,
        }

        del df, hashes, raw_labels, label_list
        gc.collect()

    zero_variance_cols = sorted(
        col for col in col_min if col in col_max and col_min[col] == col_max[col]
    )

    dedup_impact = {}
    for lbl, raw_cnt in label_counts_raw.most_common():
        kept_cnt = label_counts_dedup.get(lbl, 0)
        dropped_cnt = raw_cnt - kept_cnt
        dedup_impact[lbl] = {
            "raw_count": raw_cnt,
            "after_dedup_count": kept_cnt,
            "duplicates_removed": dropped_cnt,
            "pct_removed": round((dropped_cnt / raw_cnt) * 100, 2) if raw_cnt else 0.0,
        }

    return {
        "dataset": "CICIDS2017",
        "raw_dir": str(raw_dir),
        "total_files": len(csv_files),
        "total_rows": total_rows,
        "total_unique_rows": len(seen_hashes),
        "total_exact_duplicates": total_cross_file_dups,
        "duplicate_rate_pct": round((total_cross_file_dups / total_rows) * 100, 2) if total_rows else 0.0,
        "columns_count": len(reference_cols or []),
        "standardized_columns": reference_cols or [],
        "schema_mismatch_files": schema_mismatches,
        "non_numeric_columns_seen": sorted(object_cols_seen),
        "zero_variance_columns": zero_variance_cols,
        "total_nan_by_column": dict(total_nan.most_common()),
        "total_inf_by_column": dict(total_inf.most_common()),
        "total_negative_by_column": dict(total_neg.most_common()),
        "dedup_impact_by_label": dedup_impact,
        "files": file_reports,
    }


def _render_unsw_markdown(report: dict[str, Any]) -> str:
    lines = ["# UNSW-NB15 Raw Dataset EDA Report\n"]
    for fname, info in report["files"].items():
        lines.append(f"## File: `{fname}`")
        lines.append(
            f"- **Shape:** {info['rows']:,} rows × {info['columns']} cols ({info['size_mb']} MB, encoding=`{info['encoding']}`)"
        )
        lines.append(f"- **Categorical columns:** {info['categorical_columns']}")
        lines.append(f"- **Zero-variance columns:** {info['zero_variance_columns']}")
        lines.append(
            f"- **Duplicates:** {info['exact_duplicates_all_cols']:,} (all cols) | "
            f"{info['exact_duplicates_excluding_id']:,} (excluding `id`)"
        )
        lines.append(f"- **NaN cols:** {info['anomalies']['nan'] or 'None'}")
        lines.append(f"- **Inf cols:** {info['anomalies']['inf'] or 'None'}")
        lines.append(f"- **Negative cols:** {info['anomalies']['negative'] or 'None'}")
        lines.append(f"- **Binary `label`:** {info['binary_label_distribution']}")
        lines.append(f"- **Multiclass `attack_cat`:** {info['attack_cat_distribution']}\n")

    lines.append("## Cross-File Audit")
    lines.append("```json")
    lines.append(json.dumps(report["cross_file_audit"], indent=2))
    lines.append("```\n")
    return "\n".join(lines)


def _render_cicids_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CICIDS2017 Raw Dataset EDA Report\n",
        f"- **Files:** {report['total_files']} | **Columns:** {report['columns_count']} | **Schema mismatches:** {report['schema_mismatch_files'] or 'None'}",
        f"- **Total rows:** {report['total_rows']:,}",
        f"- **Unique rows (hash audit):** {report['total_unique_rows']:,}",
        f"- **Exact duplicates:** {report['total_exact_duplicates']:,} ({report['duplicate_rate_pct']}%)",
        f"- **Non-numeric columns seen:** {report['non_numeric_columns_seen']}",
        f"- **Zero-variance columns ({len(report['zero_variance_columns'])}):** {report['zero_variance_columns']}",
        f"- **Columns with NaN:** {report['total_nan_by_column'] or 'None'}",
        f"- **Columns with Inf:** {report['total_inf_by_column'] or 'None'}",
        f"- **Columns with Negative values:** {report['total_negative_by_column'] or 'None'}\n",
        "## Class Distribution & Deduplication Impact\n",
        "| Raw Label (`repr`) | Raw Count | After Dedup | Removed | % Removed |",
        "|---|---:|---:|---:|---:|",
    ]
    for lbl, stats in report["dedup_impact_by_label"].items():
        lines.append(
            f"| `{lbl}` | {stats['raw_count']:,} | {stats['after_dedup_count']:,} | "
            f"{stats['duplicates_removed']:,} | {stats['pct_removed']}% |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Run read-only inspection on both raw datasets and save reports to reports/eda/."""
    eda_dir = settings.resolve_path(settings.reports_dir) / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    unsw_report = inspect_unsw_nb15(settings.resolve_path(settings.unsw_raw_dir))
    (eda_dir / "unsw_nb15_eda.json").write_text(json.dumps(unsw_report, indent=2), encoding="utf-8")
    unsw_md = _render_unsw_markdown(unsw_report)
    (eda_dir / "unsw_nb15_eda.md").write_text(unsw_md, encoding="utf-8")

    cicids_report = inspect_cicids2017(settings.resolve_path(settings.cicids_raw_dir))
    (eda_dir / "cicids2017_eda.json").write_text(json.dumps(cicids_report, indent=2), encoding="utf-8")
    cicids_md = _render_cicids_markdown(cicids_report)
    (eda_dir / "cicids2017_eda.md").write_text(cicids_md, encoding="utf-8")

    logger.info("EDA reports written to %s", eda_dir)
    print("\n" + "=" * 72)
    print(unsw_md)
    print("=" * 72)
    print(cicids_md)


if __name__ == "__main__":
    main()