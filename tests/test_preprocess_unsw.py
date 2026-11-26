"""Targeted validation tests for UNSW-NB15 preprocessing outputs and leakage guarantees."""

from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd

from intrusioncopilot.config import settings
from intrusioncopilot.data.common import hash_feature_rows


def test_unsw_artifacts_and_zero_leakage() -> None:
    proc_dir = settings.resolve_path(settings.processed_dir) / "unsw_nb15"
    bundle_path = settings.resolve_path(settings.artifacts_dir) / "preprocessors" / "unsw_nb15_prep.joblib"
    report_path = settings.resolve_path(settings.reports_dir) / "preprocessing" / "unsw_nb15_report.json"

    assert bundle_path.exists(), "Missing UNSW preprocessor joblib bundle"
    assert report_path.exists(), "Missing UNSW validation report JSON"

    bundle = joblib.load(bundle_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    train = pd.read_parquet(proc_dir / "train.parquet")
    val = pd.read_parquet(proc_dir / "val.parquet")
    test = pd.read_parquet(proc_dir / "test.parquet")

    raw_cols = bundle["raw_feature_columns"]
    feat_cols = bundle["feature_columns"]

    # 1. Zero feature-hash overlap across splits
    h_tr = set(hash_feature_rows(train, raw_cols).tolist())
    h_va = set(hash_feature_rows(val, raw_cols).tolist())
    h_te = set(hash_feature_rows(test, raw_cols).tolist())
    assert len(h_tr & h_va) == 0
    assert len(h_tr & h_te) == 0
    assert len(h_va & h_te) == 0

    # 2. Target alignment & complete class coverage
    for df in (train, val, test):
        assert "id" not in df.columns
        assert ((df["attack_cat"] != "Normal").astype(np.int8) == df["y_binary"]).all()
        assert df["y_multiclass"].nunique() == 10

    # 3. Preprocessors handle unseen categorical values safely without NaN/Inf
    sample = test[feat_cols].head(5).copy()
    sample.loc[sample.index[0], "proto"] = "unseen_proto_xyz"
    sample.loc[sample.index[0], "service"] = "unseen_svc_xyz"
    sample.loc[sample.index[0], "state"] = "UNSEEN_STATE"

    X_tree = bundle["tree_preprocessor"].transform(sample)
    X_ae = bundle["ae_preprocessor"].transform(sample)
    assert np.isfinite(X_tree).all()
    assert np.isfinite(X_ae).all()
    assert X_tree.shape == (5, report["preprocessors"]["tree_output_dim"])
    assert X_ae.shape == (5, report["preprocessors"]["ae_output_dim"])