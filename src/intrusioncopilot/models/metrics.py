"""Evaluation metrics, false-positive analysis, and latency benchmarking utilities."""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from intrusioncopilot.config import settings


def load_training_config() -> dict[str, Any]:
    """Load supervised training configuration from configs/training.json."""
    cfg_path = settings.resolve_path(settings.configs_dir) / "training.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing training config: {cfg_path}")
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def transform_features(
    df: pd.DataFrame,
    bundle: dict[str, Any],
    mode: str = "tree",
) -> np.ndarray:
    """Transform a processed DataFrame using the matching preprocessor without dtype/name warnings."""
    feat_cols = bundle["feature_columns"]
    prep = bundle["tree_preprocessor"] if mode == "tree" else bundle["ae_preprocessor"]
    if bundle.get("categorical_columns"):
        X_out = prep.transform(df[feat_cols])
    else:
        X_out = prep.transform(df[feat_cols].to_numpy(dtype=np.float32, copy=False))
    return np.asarray(X_out, dtype=np.float32)


def compute_dampened_sample_weights(
    y: np.ndarray,
    power: float = 0.5,
    cap: float = 50.0,
) -> tuple[np.ndarray, dict[int, float]]:
    """Compute square-root-dampened, capped inverse-frequency sample weights for multiclass imbalance."""
    classes, counts = np.unique(y, return_counts=True)
    n_total = len(y)
    n_classes = len(classes)

    class_weights: dict[int, float] = {}
    for cls, cnt in zip(classes.tolist(), counts.tolist()):
        raw_w = n_total / (n_classes * cnt)
        dampened_w = min(float(raw_w**power), float(cap))
        class_weights[int(cls)] = round(dampened_w, 4)

    sample_weights = np.array([class_weights[int(lbl)] for lbl in y], dtype=np.float32)
    return sample_weights, class_weights


def compute_operational_fpr(y_true: np.ndarray, y_pred: np.ndarray, benign_id: int = 0) -> dict[str, Any]:
    """Compute operational SOC False Positive Rate: Benign (0) traffic predicted as Malicious (!= 0)."""
    is_benign_true = y_true == benign_id
    is_mal_pred = y_pred != benign_id

    tn = int(np.sum(is_benign_true & ~is_mal_pred))
    fp = int(np.sum(is_benign_true & is_mal_pred))
    fn = int(np.sum(~is_benign_true & ~is_mal_pred))
    tp = int(np.sum(~is_benign_true & is_mal_pred))

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return {
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "false_positive_rate": round(float(fpr), 6),
        "false_negative_rate": round(float(fnr), 6),
    }


def analyze_false_positives(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    task: str,
    id_to_label: dict[int, str],
    benign_id: int = 0,
) -> dict[str, Any]:
    """Analyze Benign rows falsely flagged as Malicious (confidence distribution, predicted attacks, context)."""
    fp_mask = (y_true == benign_id) & (y_pred != benign_id)
    fp_count = int(fp_mask.sum())
    total_benign = int((y_true == benign_id).sum())

    if fp_count == 0:
        return {
            "total_benign": total_benign,
            "false_positives": 0,
            "false_positive_rate": 0.0,
            "confidence_quantiles": {},
            "predicted_attack_breakdown": {},
            "context_breakdown": {},
        }

    if task == "binary":
        fp_conf = y_proba[fp_mask, 1]
    else:
        fp_conf = np.max(y_proba[fp_mask], axis=1)

    conf_quantiles = {
        "min": round(float(np.min(fp_conf)), 4),
        "median": round(float(np.median(fp_conf)), 4),
        "p90": round(float(np.quantile(fp_conf, 0.90)), 4),
        "max": round(float(np.max(fp_conf)), 4),
        "high_confidence_ge_0_80": int(np.sum(fp_conf >= 0.80)),
    }

    pred_breakdown: dict[str, int] = {}
    if task == "multiclass":
        pred_ids, pred_counts = np.unique(y_pred[fp_mask], return_counts=True)
        pairs = sorted(zip(pred_ids.tolist(), pred_counts.tolist()), key=lambda x: x[1], reverse=True)
        pred_breakdown = {id_to_label.get(cid, str(cid)): int(cnt) for cid, cnt in pairs}

    # Summarize dataset-specific network context attributes on falsely flagged Benign flows
    fp_df = df.loc[fp_mask]
    context_breakdown: dict[str, dict[str, int]] = {}
    candidate_cols = [c for c in ("service", "proto", "state", "Destination Port") if c in fp_df.columns]
    for col in candidate_cols:
        vc = fp_df[col].value_counts().head(8)
        context_breakdown[col] = {
            str(int(k) if isinstance(k, (float, np.floating)) and float(k).is_integer() else k): int(v)
            for k, v in vc.items()
        }

    return {
        "total_benign": total_benign,
        "false_positives": fp_count,
        "false_positive_rate": round(fp_count / total_benign, 6) if total_benign > 0 else 0.0,
        "confidence_quantiles": conf_quantiles,
        "predicted_attack_breakdown": pred_breakdown,
        "context_breakdown": context_breakdown,
    }


def evaluate_predictions(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    task: str,
    id_to_label: dict[int, str],
) -> dict[str, Any]:
    """Compute comprehensive binary or multiclass classification metrics plus FP error analysis."""
    labels_sorted = sorted(id_to_label.keys())
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_sorted, zero_division=0
    )
    per_class: dict[str, dict[str, Any]] = {}
    for idx, cid in enumerate(labels_sorted):
        per_class[id_to_label[cid]] = {
            "class_id": int(cid),
            "precision": round(float(prec[idx]), 4),
            "recall": round(float(rec[idx]), 4),
            "f1": round(float(f1[idx]), 4),
            "support": int(sup[idx]),
        }

    op_fpr = compute_operational_fpr(y_true, y_pred, benign_id=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted).tolist()
    fp_analysis = analyze_false_positives(df, y_true, y_pred, y_proba, task, id_to_label, benign_id=0)

    summary: dict[str, Any] = {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "operational_fpr": op_fpr["false_positive_rate"],
        "operational_fnr": op_fpr["false_negative_rate"],
        "operational_confusion": op_fpr,
    }

    if task == "binary":
        pos_prob = y_proba[:, 1]
        summary["malicious_precision"] = per_class[id_to_label[1]]["precision"]
        summary["malicious_recall"] = per_class[id_to_label[1]]["recall"]
        summary["malicious_f1"] = per_class[id_to_label[1]]["f1"]
        summary["roc_auc"] = round(float(roc_auc_score(y_true, pos_prob)), 5)
        summary["pr_auc"] = round(float(average_precision_score(y_true, pos_prob)), 5)

    return {
        "summary": summary,
        "per_class": per_class,
        "confusion_matrix": {"labels": [id_to_label[c] for c in labels_sorted], "matrix": cm},
        "false_positive_analysis": fp_analysis,
    }


def benchmark_inference_latency(
    model: Any,
    X_test: np.ndarray,
    single_runs: int = 100,
    batch_size: int = 10000,
) -> dict[str, float]:
    """Measure single-flow (real-time API) and batch CPU inference latency in milliseconds."""
    n_samples = len(X_test)
    n_single = min(single_runs, n_samples)
    n_batch = min(batch_size, n_samples)

    # Warm-up call
    _ = model.predict_proba(X_test[:1])

    single_times_ms: list[float] = []
    for i in range(n_single):
        row = X_test[i : i + 1]
        t0 = time.perf_counter()
        _ = model.predict_proba(row)
        single_times_ms.append((time.perf_counter() - t0) * 1000.0)

    X_batch = X_test[:n_batch]
    t0 = time.perf_counter()
    _ = model.predict_proba(X_batch)
    batch_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "single_flow_runs": n_single,
        "single_flow_median_ms": round(float(np.median(single_times_ms)), 3),
        "single_flow_p95_ms": round(float(np.quantile(single_times_ms, 0.95)), 3),
        "batch_size": n_batch,
        "batch_total_ms": round(float(batch_ms), 2),
        "ms_per_1000_flows": round(float(batch_ms * 1000.0 / n_batch), 3),
    }


def extract_top_feature_importances(
    model: Any,
    feature_names: list[str],
    top_k: int = 15,
) -> list[dict[str, Any]]:
    """Return the top-k most important features from a tree-based classifier."""
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []
    arr = np.asarray(importances, dtype=np.float64)
    order = np.argsort(arr)[::-1][:top_k]
    return [
        {"feature": feature_names[int(idx)], "importance": round(float(arr[int(idx)]), 5)}
        for idx in order
    ]