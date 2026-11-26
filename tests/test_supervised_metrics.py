"""Unit tests for supervised evaluation metrics, dampened weights, and UNSW-NB15 model artifacts."""

from __future__ import annotations

import json
import numpy as np
import pandas as pd

from intrusioncopilot.config import settings
from intrusioncopilot.models.metrics import (
    compute_dampened_sample_weights,
    compute_operational_fpr,
    evaluate_predictions,
)


def test_operational_fpr_multiclass_collapse() -> None:
    # 4 Benign (0): 1 predicted as class 2 (FP), 3 predicted as 0 (TN) -> FPR = 0.25
    # 2 Malicious (1, 2): 1 predicted as 0 (FN), 1 predicted as 1 (TP) -> FNR = 0.50
    y_true = np.array([0, 0, 0, 0, 1, 2])
    y_pred = np.array([0, 0, 0, 2, 0, 1])
    res = compute_operational_fpr(y_true, y_pred, benign_id=0)
    assert res["true_negatives"] == 3
    assert res["false_positives"] == 1
    assert res["false_negatives"] == 1
    assert res["true_positives"] == 1
    assert res["false_positive_rate"] == 0.25
    assert res["false_negative_rate"] == 0.5


def test_dampened_sample_weights_respect_cap() -> None:
    y = np.array([0] * 10000 + [1] * 2)
    sample_w, class_w = compute_dampened_sample_weights(y, power=0.5, cap=25.0)
    assert len(sample_w) == len(y)
    assert class_w[1] == 25.0
    assert class_w[0] < 1.0


def test_false_positive_analysis_tracks_context() -> None:
    df = pd.DataFrame({"service": ["http", "dns", "http", "ssh"], "proto": ["tcp", "udp", "tcp", "tcp"]})
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([2, 0, 1, 2])
    y_proba = np.array(
        [
            [0.1, 0.05, 0.85],
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.2, 0.75],
        ],
        dtype=np.float32,
    )
    res = evaluate_predictions(df, y_true, y_pred, y_proba, "multiclass", {0: "Normal", 1: "DoS", 2: "Fuzzers"})
    fp = res["false_positive_analysis"]
    assert fp["false_positives"] == 1
    assert fp["predicted_attack_breakdown"] == {"Fuzzers": 1}
    assert fp["context_breakdown"]["service"] == {"http": 1}


def test_unsw_supervised_artifacts_and_report_exist() -> None:
    model_dir = settings.resolve_path(settings.artifacts_dir) / "models" / "unsw_nb15"
    report_path = settings.resolve_path(settings.reports_dir) / "evaluation" / "unsw_nb15_supervised_report.json"

    assert report_path.exists(), "Run supervised training on unsw_nb15 first"
    for fname in ("binary_rf.joblib", "binary_xgb.ubj", "multiclass_rf.joblib", "multiclass_xgb.ubj", "manifest.json"):
        assert (model_dir / fname).exists(), f"Missing artifact: {fname}"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    for task in ("binary", "multiclass"):
        assert report["tasks"][task]["selection"]["selected_model"] in ("rf", "xgb")
        for m in ("rf", "xgb"):
            summary = report["tasks"][task]["models"][m]["test_metrics"]["summary"]
            assert 0.0 <= summary["operational_fpr"] <= 1.0
            assert 0.0 < summary["macro_f1"] <= 1.0



def test_cicids_supervised_artifacts_and_report_exist() -> None:
    model_dir = settings.resolve_path(settings.artifacts_dir) / "models" / "cicids2017"
    report_path = settings.resolve_path(settings.reports_dir) / "evaluation" / "cicids2017_supervised_report.json"

    assert report_path.exists(), "Run supervised training on cicids2017 first"
    for fname in ("binary_rf.joblib", "binary_xgb.ubj", "multiclass_rf.joblib", "multiclass_xgb.ubj", "manifest.json"):
        assert (model_dir / fname).exists(), f"Missing artifact: {fname}"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    for task in ("binary", "multiclass"):
        assert report["tasks"][task]["selection"]["selected_model"] in ("rf", "xgb")
        for m in ("rf", "xgb"):
            summary = report["tasks"][task]["models"][m]["test_metrics"]["summary"]
            assert 0.0 <= summary["operational_fpr"] <= 1.0
            assert 0.0 < summary["macro_f1"] <= 1.0