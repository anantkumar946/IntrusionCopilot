"""Supervised ML training & evaluation pipeline (Random Forest & XGBoost) for IntrusionCopilot.

Run via:
    uv run python -m intrusioncopilot.models.supervised --dataset unsw_nb15
    uv run python -m intrusioncopilot.models.supervised --dataset cicids2017
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

from intrusioncopilot.config import settings
from intrusioncopilot.models.metrics import (
    benchmark_inference_latency,
    compute_dampened_sample_weights,
    evaluate_predictions,
    extract_top_feature_importances,
    load_training_config,
    transform_features,
)
from intrusioncopilot.utils.logging import get_logger

logger = get_logger(__name__)


def _train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    rf_cfg: dict[str, Any],
    seed: int,
) -> tuple[RandomForestClassifier, float]:
    """Train Random Forest classifier with class-balanced subsampling."""
    params = dict(rf_cfg)
    params["random_state"] = seed
    clf = RandomForestClassifier(**params)
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_sec = round(time.perf_counter() - t0, 2)
    return clf, train_sec


def _train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task: str,
    n_classes: int,
    xgb_cfg: dict[str, Any],
    seed: int,
    weight_power: float,
    weight_cap: float,
) -> tuple[xgb.XGBClassifier, float, dict[str, Any]]:
    """Train XGBoost classifier on GPU (with CPU fallback) using validation early stopping."""
    params = dict(xgb_cfg)
    params["random_state"] = seed
    params["n_jobs"] = -1

    weight_meta: dict[str, Any] = {}
    sample_weight: np.ndarray | None = None

    if task == "binary":
        n_neg = int(np.sum(y_train == 0))
        n_pos = int(np.sum(y_train == 1))
        spw = round(n_neg / max(n_pos, 1), 4)
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss"
        params["scale_pos_weight"] = spw
        weight_meta["scale_pos_weight"] = spw
    else:
        params["objective"] = "multi:softprob"
        params["num_class"] = n_classes
        params["eval_metric"] = "mlogloss"
        sample_weight, class_weights = compute_dampened_sample_weights(
            y_train, power=weight_power, cap=weight_cap
        )
        weight_meta["dampened_class_weights"] = class_weights

    requested_device = params.get("device", "cpu")
    used_device = requested_device

    t0 = time.perf_counter()
    try:
        clf = xgb.XGBClassifier(**params)
        clf.fit(X_train, y_train, sample_weight=sample_weight, eval_set=[(X_val, y_val)], verbose=False)
    except Exception as exc:
        if requested_device != "cpu":
            logger.warning("XGBoost device=%s failed (%s); falling back to CPU.", requested_device, exc)
            params["device"] = "cpu"
            used_device = "cpu"
            t0 = time.perf_counter()
            clf = xgb.XGBClassifier(**params)
            clf.fit(X_train, y_train, sample_weight=sample_weight, eval_set=[(X_val, y_val)], verbose=False)
        else:
            raise

    train_sec = round(time.perf_counter() - t0, 2)

    # Switch booster to CPU for deterministic, portable single-flow & batch serving latency
    clf.set_params(device="cpu")
    clf.get_booster().set_param({"device": "cpu"})

    weight_meta["train_device"] = used_device
    weight_meta["best_iteration"] = int(getattr(clf, "best_iteration", params["n_estimators"] - 1))
    return clf, train_sec, weight_meta


def _save_model_artifact(model: Any, model_type: str, out_dir: Path, task: str) -> Path:
    """Save trained model in compact native format (.joblib for RF, .ubj for XGBoost)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if model_type == "rf":
        path = out_dir / f"{task}_rf.joblib"
        joblib.dump(model, path, compress=3)
    else:
        path = out_dir / f"{task}_xgb.ubj"
        model.save_model(path)
    return path


def _select_winner(rf_res: dict[str, Any], xgb_res: dict[str, Any]) -> dict[str, Any]:
    """Select best model using Macro-F1 with operational FPR as tie-breaker/guardrail."""
    rf_s = rf_res["test_metrics"]["summary"]
    xgb_s = xgb_res["test_metrics"]["summary"]

    # Primary criterion: Macro-F1; secondary: lower operational FPR
    rf_score = (rf_s["macro_f1"], -rf_s["operational_fpr"])
    xgb_score = (xgb_s["macro_f1"], -xgb_s["operational_fpr"])
    winner = "xgb" if xgb_score >= rf_score else "rf"
    return {
        "selected_model": winner,
        "criterion": "Highest test Macro-F1 with operational Benign FPR tie-breaker",
        "rf_macro_f1": rf_s["macro_f1"],
        "rf_fpr": rf_s["operational_fpr"],
        "xgb_macro_f1": xgb_s["macro_f1"],
        "xgb_fpr": xgb_s["operational_fpr"],
    }


def _render_markdown(report: dict[str, Any]) -> str:
    ds = report["dataset"].upper()
    lines = [
        f"# {ds} — Supervised ML Evaluation Report (Random Forest vs. XGBoost)\n",
        f"- **Random Seed:** `{report['random_seed']}`",
        f"- **Split Sizes:** Train `{report['split_sizes']['train']:,}` | "
        f"Val `{report['split_sizes']['val']:,}` | Test `{report['split_sizes']['test']:,}`",
        f"- **Tree Input Features:** `{report['n_features']}`\n",
        "## Test-Set Model Comparison\n",
        "| Task | Model | Accuracy | Macro-F1 | Weighted-F1 | Op. Benign FPR | ROC-AUC | Train (s) | Single Flow (ms) | Batch (ms/1k) | Artifact (MB) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task in ("binary", "multiclass"):
        for mname in ("rf", "xgb"):
            m = report["tasks"][task]["models"][mname]
            s = m["test_metrics"]["summary"]
            lat = m["latency"]
            roc = f"{s['roc_auc']:.4f}" if "roc_auc" in s else "—"
            lines.append(
                f"| `{task}` | `{mname.upper()}` | {s['accuracy']:.4f} | **{s['macro_f1']:.4f}** | "
                f"{s['weighted_f1']:.4f} | **{s['operational_fpr']*100:.2f}%** | {roc} | "
                f"{m['train_seconds']:.1f} | {lat['single_flow_median_ms']:.2f} | "
                f"{lat['ms_per_1000_flows']:.2f} | {m['artifact_size_mb']:.2f} |"
            )

    lines.append("\n## Model Selection Summary\n")
    for task in ("binary", "multiclass"):
        sel = report["tasks"][task]["selection"]
        lines.append(
            f"- **`{task}` Winner:** `{sel['selected_model'].upper()}` "
            f"(XGB Macro-F1={sel['xgb_macro_f1']:.4f}, FPR={sel['xgb_fpr']*100:.2f}% vs. "
            f"RF Macro-F1={sel['rf_macro_f1']:.4f}, FPR={sel['rf_fpr']*100:.2f}%)"
        )

    # Multiclass per-class comparison table
    lines.append("\n## Multiclass Per-Class Test Performance (RF vs. XGBoost)\n")
    lines.append("| Attack Class | Support | RF Prec | RF Rec | RF F1 | XGB Prec | XGB Rec | XGB F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    rf_pc = report["tasks"]["multiclass"]["models"]["rf"]["test_metrics"]["per_class"]
    xgb_pc = report["tasks"]["multiclass"]["models"]["xgb"]["test_metrics"]["per_class"]
    for cls_name in rf_pc:
        r = rf_pc[cls_name]
        x = xgb_pc[cls_name]
        lines.append(
            f"| `{cls_name}` | {r['support']:,} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | "
            f"{x['precision']:.4f} | {x['recall']:.4f} | {x['f1']:.4f} |"
        )

    # False-positive error analysis for selected multiclass model
    mc_winner = report["tasks"]["multiclass"]["selection"]["selected_model"]
    fp_info = report["tasks"]["multiclass"]["models"][mc_winner]["test_metrics"]["false_positive_analysis"]
    lines.extend(
        [
            f"\n## False-Positive Error Analysis (`multiclass` `{mc_winner.upper()}` on Held-Out Test)\n",
            f"- **Total Benign Test Flows:** `{fp_info['total_benign']:,}`",
            f"- **False Positives:** `{fp_info['false_positives']:,}` (`{fp_info['false_positive_rate']*100:.2f}%` FPR)",
            f"- **FP Confidence Quantiles:** `{fp_info['confidence_quantiles']}`",
            f"- **Benign Traffic Misclassified As:** `{fp_info['predicted_attack_breakdown']}`",
            f"- **Top Network Context Attributes Among FPs:** `{fp_info['context_breakdown']}`\n",
        ]
    )
    return "\n".join(lines)


def run_supervised_for_dataset(dataset: str) -> dict[str, Any]:
    """Train, evaluate, benchmark, and persist RF & XGBoost models for a single dataset."""
    cfg = load_training_config()
    seed = int(cfg["random_seed"])
    ds_cfg = cfg[dataset]
    lat_cfg = cfg["latency_benchmark"]

    proc_dir = settings.resolve_path(settings.processed_dir) / dataset
    prep_path = settings.resolve_path(settings.artifacts_dir) / "preprocessors" / f"{dataset}_prep.joblib"
    model_dir = settings.resolve_path(settings.artifacts_dir) / "models" / dataset
    eval_dir = settings.resolve_path(settings.reports_dir) / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Phase 2 Supervised Training: %s ===", dataset.upper())
    bundle = joblib.load(prep_path)
    feature_names = list(bundle["tree_feature_names"])

    train_df = pd.read_parquet(proc_dir / "train.parquet")
    val_df = pd.read_parquet(proc_dir / "val.parquet")
    test_df = pd.read_parquet(proc_dir / "test.parquet")

    logger.info("Transforming train/val/test feature matrices (%d features)...", len(feature_names))
    X_train = transform_features(train_df, bundle, mode="tree")
    X_val = transform_features(val_df, bundle, mode="tree")
    X_test = transform_features(test_df, bundle, mode="tree")

    y_bin_train = train_df["y_binary"].to_numpy(dtype=np.int32)
    y_bin_val = val_df["y_binary"].to_numpy(dtype=np.int32)
    y_bin_test = test_df["y_binary"].to_numpy(dtype=np.int32)

    y_mc_train = train_df["y_multiclass"].to_numpy(dtype=np.int32)
    y_mc_val = val_df["y_multiclass"].to_numpy(dtype=np.int32)
    y_mc_test = test_df["y_multiclass"].to_numpy(dtype=np.int32)

    # Free heavy feature columns in train_df/val_df now that matrices are ready (keep test_df context for FP analysis)
    split_sizes = {"train": len(train_df), "val": len(val_df), "test": len(test_df)}
    del train_df, val_df
    gc.collect()

    benign_name = bundle["id_to_attack_cat"]["0"]
    binary_id_to_label = {0: benign_name, 1: "Malicious"}
    mc_id_to_label = {int(k): v for k, v in bundle["id_to_attack_cat"].items()}

    tasks_spec = [
        ("binary", y_bin_train, y_bin_val, y_bin_test, binary_id_to_label),
        ("multiclass", y_mc_train, y_mc_val, y_mc_test, mc_id_to_label),
    ]

    task_reports: dict[str, Any] = {}
    manifest_models: dict[str, Any] = {}

    for task_name, y_tr, y_va, y_te, id_to_lbl in tasks_spec:
        n_classes = len(id_to_lbl)
        models_out: dict[str, Any] = {}

        # 1. Random Forest
        logger.info("[%s | %s] Training Random Forest...", dataset, task_name)
        rf_clf, rf_sec = _train_random_forest(X_train, y_tr, ds_cfg["rf"], seed)
        rf_val_proba = rf_clf.predict_proba(X_val)
        rf_val_pred = np.argmax(rf_val_proba, axis=1)
        rf_test_proba = rf_clf.predict_proba(X_test)
        rf_test_pred = np.argmax(rf_test_proba, axis=1)

        rf_val_f1 = round(float(evaluate_predictions(
            pd.DataFrame(index=range(len(y_va))), y_va, rf_val_pred, rf_val_proba, task_name, id_to_lbl
        )["summary"]["macro_f1"]), 4)
        rf_test_metrics = evaluate_predictions(test_df, y_te, rf_test_pred, rf_test_proba, task_name, id_to_lbl)
        rf_latency = benchmark_inference_latency(
            rf_clf, X_test, single_runs=lat_cfg["single_flow_runs"], batch_size=lat_cfg["batch_size"]
        )
        rf_path = _save_model_artifact(rf_clf, "rf", model_dir, task_name)
        rf_top_feats = extract_top_feature_importances(rf_clf, feature_names, top_k=15)

        models_out["rf"] = {
            "model_type": "RandomForestClassifier",
            "train_seconds": rf_sec,
            "val_macro_f1": rf_val_f1,
            "test_metrics": rf_test_metrics,
            "latency": rf_latency,
            "top_features": rf_top_feats,
            "artifact_path": str(rf_path.relative_to(settings.project_root)),
            "artifact_size_mb": round(rf_path.stat().st_size / (1024 * 1024), 2),
        }
        logger.info(
            "[%s | %s] RF done in %.1fs | Test Macro-F1=%.4f | FPR=%.2f%%",
            dataset, task_name, rf_sec,
            rf_test_metrics["summary"]["macro_f1"],
            rf_test_metrics["summary"]["operational_fpr"] * 100,
        )
        del rf_clf, rf_val_proba, rf_test_proba
        gc.collect()

        # 2. XGBoost
        logger.info("[%s | %s] Training XGBoost (device=%s)...", dataset, task_name, ds_cfg["xgb"]["device"])
        xgb_clf, xgb_sec, xgb_meta = _train_xgboost(
            X_train, y_tr, X_val, y_va, task_name, n_classes, ds_cfg["xgb"], seed,
            weight_power=float(cfg["multiclass_weight_power"]),
            weight_cap=float(cfg["multiclass_weight_cap"]),
        )
        xgb_val_proba = xgb_clf.predict_proba(X_val)
        xgb_val_pred = np.argmax(xgb_val_proba, axis=1)
        xgb_test_proba = xgb_clf.predict_proba(X_test)
        xgb_test_pred = np.argmax(xgb_test_proba, axis=1)

        xgb_val_f1 = round(float(evaluate_predictions(
            pd.DataFrame(index=range(len(y_va))), y_va, xgb_val_pred, xgb_val_proba, task_name, id_to_lbl
        )["summary"]["macro_f1"]), 4)
        xgb_test_metrics = evaluate_predictions(test_df, y_te, xgb_test_pred, xgb_test_proba, task_name, id_to_lbl)
        xgb_latency = benchmark_inference_latency(
            xgb_clf, X_test, single_runs=lat_cfg["single_flow_runs"], batch_size=lat_cfg["batch_size"]
        )
        xgb_path = _save_model_artifact(xgb_clf, "xgb", model_dir, task_name)
        xgb_top_feats = extract_top_feature_importances(xgb_clf, feature_names, top_k=15)

        models_out["xgb"] = {
            "model_type": "XGBClassifier",
            "train_seconds": xgb_sec,
            "training_meta": xgb_meta,
            "val_macro_f1": xgb_val_f1,
            "test_metrics": xgb_test_metrics,
            "latency": xgb_latency,
            "top_features": xgb_top_feats,
            "artifact_path": str(xgb_path.relative_to(settings.project_root)),
            "artifact_size_mb": round(xgb_path.stat().st_size / (1024 * 1024), 2),
        }
        logger.info(
            "[%s | %s] XGB done in %.1fs (best_iter=%d) | Test Macro-F1=%.4f | FPR=%.2f%%",
            dataset, task_name, xgb_sec, xgb_meta["best_iteration"],
            xgb_test_metrics["summary"]["macro_f1"],
            xgb_test_metrics["summary"]["operational_fpr"] * 100,
        )
        del xgb_clf, xgb_val_proba, xgb_test_proba
        gc.collect()

        selection = _select_winner(models_out["rf"], models_out["xgb"])
        task_reports[task_name] = {"models": models_out, "selection": selection}
        manifest_models[task_name] = {
            "selected_model": selection["selected_model"],
            "rf_artifact": models_out["rf"]["artifact_path"],
            "xgb_artifact": models_out["xgb"]["artifact_path"],
        }

    report: dict[str, Any] = {
        "dataset": dataset,
        "random_seed": seed,
        "split_sizes": split_sizes,
        "n_features": len(feature_names),
        "preprocessor_path": str(prep_path.relative_to(settings.project_root)),
        "config": ds_cfg,
        "tasks": task_reports,
    }

    # Save manifest inside artifacts/models/<dataset>/ and full reports inside reports/evaluation/
    manifest_path = model_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "preprocessor_path": report["preprocessor_path"],
                "models": manifest_models,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    json_path = eval_dir / f"{dataset}_supervised_report.json"
    md_path = eval_dir / f"{dataset}_supervised_report.md"
    md_text = _render_markdown(report)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(md_text, encoding="utf-8")

    logger.info("Saved %s supervised evaluation report to %s and %s", dataset, json_path, md_path)
    print("\n" + md_text)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train & evaluate RF and XGBoost intrusion detectors.")
    parser.add_argument(
        "--dataset",
        choices=("unsw_nb15", "cicids2017", "all"),
        default="unsw_nb15",
        help="Dataset to train and evaluate.",
    )
    args = parser.parse_args()
    targets = ["unsw_nb15", "cicids2017"] if args.dataset == "all" else [args.dataset]
    for ds in targets:
        run_supervised_for_dataset(ds)


if __name__ == "__main__":
    main()