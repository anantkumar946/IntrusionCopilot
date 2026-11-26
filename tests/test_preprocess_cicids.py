"""Targeted validation tests for CICIDS2017 preprocessing outputs and leakage guarantees."""

from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd

from intrusioncopilot.config import settings
from intrusioncopilot.data.common import hash_feature_rows


def test_cicids_artifacts_and_zero_leakage() -> None:
    proc_dir = settings.resolve_path(settings.processed_dir) / "cicids2017"
    bundle_path = settings.resolve_path(settings.artifacts_dir) / "preprocessors" / "cicids2017_prep.joblib"
    report_path = settings.resolve_path(settings.reports_dir) / "preprocessing" / "cicids2017_report.json"

    assert bundle_path.exists(), "Missing CICIDS2017 preprocessor joblib bundle"
    assert report_path.exists(), "Missing CICIDS2017 validation report JSON"

    bundle = joblib.load(bundle_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    raw_cols = bundle["raw_feature_columns"]
    feat_cols = bundle["feature_columns"]

    train = pd.read_parquet(proc_dir / "train.parquet")
    val = pd.read_parquet(proc_dir / "val.parquet")
    test = pd.read_parquet(proc_dir / "test.parquet")

    # 1. Zero feature-hash overlap across splits
    h_tr = set(hash_feature_rows(train, raw_cols).tolist())
    h_va = set(hash_feature_rows(val, raw_cols).tolist())
    h_te = set(hash_feature_rows(test, raw_cols).tolist())
    assert len(h_tr & h_va) == 0
    assert len(h_tr & h_te) == 0
    assert len(h_va & h_te) == 0

    # 2. Target alignment, ASCII label hygiene, and sentinel preservation
    for df in (train, val, test):
        assert "Fwd Header Length.1" not in df.columns
        assert ((df["attack_cat"] != "BENIGN").astype(np.int8) == df["y_binary"]).all()
        assert df["y_multiclass"].nunique() == len(bundle["attack_cat_to_id"])
        for lbl in df["attack_cat"].unique():
            assert lbl.isascii(), f"Non-ASCII label leaked: {lbl!r}"

    # Verify -1 TCP window sentinel was preserved
    assert (train["Init_Win_bytes_forward"] == -1).any()
    assert (train["Init_Win_bytes_backward"] == -1).any()

    # 3. Preprocessors handle NaN safely and output finite float32 matrices
    sample = test[feat_cols].head(10).to_numpy(dtype=np.float32, copy=True)
    sample[0, 0] = np.nan
    X_tree = bundle["tree_preprocessor"].transform(sample)
    X_ae = bundle["ae_preprocessor"].transform(sample)
    assert np.isfinite(X_tree).all()
    assert np.isfinite(X_ae).all()
    assert X_tree.shape == (10, report["preprocessors"]["tree_output_dim"])
    assert X_ae.shape == (10, report["preprocessors"]["ae_output_dim"])