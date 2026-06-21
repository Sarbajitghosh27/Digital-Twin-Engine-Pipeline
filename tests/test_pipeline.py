"""
tests/test_pipeline.py — Pipeline Integrity Pytest Suite

Tests:
  1. test_leakage_correlation: Ensures no feature has a near-perfect deterministic
     correlation with the cycle index (data leakage proxy test).
  2. test_data_shapes: Verifies that the sliding window outputs from the AE and
     LSTM have consistent dimensions matching the config.
"""
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np

from aiml_core.config import CONFIG
from aiml_core.data_loader import (
    DatasetManager, normalize_regimes, prepare_sliding_windows, CMAPSS_SENSORS
)


# ─── Fixture: shared synthetic dataset ────────────────────────────────────────

@pytest.fixture(scope="module")
def fd001_datasets():
    """Loads or generates FD001 train/test data once per module."""
    dm = DatasetManager(data_root="data")
    train_df, test_df = dm.get_dataset("FD001")
    train_df = train_df.ffill().bfill()
    test_df = test_df.ffill().bfill()
    return dm, train_df, test_df


# ─── Test 1: No deterministic data leakage ────────────────────────────────────

def test_leakage_correlation(fd001_datasets):
    """
    Checks that no CMAPSS sensor feature has a Pearson |r| > 0.98 with the
    raw cycle index.  A perfect or near-perfect monotonic correlation would
    indicate that the model can trivially learn RUL from a direct leakage
    signal (e.g. an artificial Vibration or Efficiency column derived from cycle).
    """
    _, train_df, _ = fd001_datasets

    LEAKAGE_THRESHOLD = 0.98  # |r| above this is suspicious

    leaky_sensors = []
    for sensor in CMAPSS_SENSORS:
        if sensor not in train_df.columns:
            continue
        # Compute Pearson r between sensor and cycle *per engine* and take max
        max_abs_corr = 0.0
        for eid in train_df["engine_id"].unique():
            eng = train_df[train_df["engine_id"] == eid].sort_values("cycle")
            if len(eng) < 5:
                continue
            cycles = eng["cycle"].values.astype(float)
            vals   = eng[sensor].values.astype(float)
            std_c  = cycles.std()
            std_v  = vals.std()
            if std_c < 1e-6 or std_v < 1e-6:
                continue
            r = float(np.corrcoef(cycles, vals)[0, 1])
            max_abs_corr = max(max_abs_corr, abs(r))

        if max_abs_corr > LEAKAGE_THRESHOLD:
            leaky_sensors.append((sensor, round(max_abs_corr, 4)))

    assert len(leaky_sensors) == 0, (
        f"Potential data leakage detected in sensors: {leaky_sensors}. "
        f"Each has |Pearson r| > {LEAKAGE_THRESHOLD} with cycle index."
    )


# ─── Test 2: Consistent dimension shapes ──────────────────────────────────────

def test_data_shapes(fd001_datasets):
    """
    Verifies that sliding window outputs match config dimensions:
      - AE input:   (N, window_size, n_sensors) where n_sensors == len(CMAPSS_SENSORS)
      - LSTM input: (M, window_size, 1)          (Health Index sequences)
    Also checks that the window count N > 0 for FD001 (enough data exists).
    """
    _, train_df, _ = fd001_datasets

    window_size = CONFIG["window_size"]
    n_sensors   = len(CMAPSS_SENSORS)

    # Ensure all sensor columns exist
    for s in CMAPSS_SENSORS:
        assert s in train_df.columns, (
            f"Expected sensor '{s}' in training DataFrame but it was missing. "
            "Check data_loader.py sensor mapping."
        )

    norm_df = normalize_regimes(
        train_df, CMAPSS_SENSORS,
        regime_col="regime" if "regime" in train_df.columns else "Setting1"
    )

    # ── AE windows (raw sensor sequences) ──────────────────────────────────
    X_ae, _ = prepare_sliding_windows(norm_df, CMAPSS_SENSORS, window_size=window_size)

    assert X_ae.ndim == 3, (
        f"AE input should be 3-D (N, window, sensors), got shape {X_ae.shape}"
    )
    assert X_ae.shape[0] > 0, (
        "No sliding windows were generated for AE — training set may be too small."
    )
    assert X_ae.shape[1] == window_size, (
        f"AE window length mismatch: expected {window_size}, got {X_ae.shape[1]}"
    )
    assert X_ae.shape[2] == n_sensors, (
        f"AE feature dimension mismatch: expected {n_sensors} sensors, "
        f"got {X_ae.shape[2]}. Check CMAPSS_SENSORS list."
    )

    # ── Simulate HI column (constant healthy = 100.0 for shape test only) ──
    norm_df["HI"] = 100.0

    X_lstm, Y_lstm = prepare_sliding_windows(norm_df, ["HI"], window_size=window_size)

    assert X_lstm.ndim == 3, (
        f"LSTM input should be 3-D (M, window, 1), got shape {X_lstm.shape}"
    )
    assert X_lstm.shape[0] > 0, (
        "No sliding windows generated for LSTM HI sequences."
    )
    assert X_lstm.shape[1] == window_size, (
        f"LSTM window length mismatch: expected {window_size}, got {X_lstm.shape[1]}"
    )
    assert X_lstm.shape[2] == 1, (
        f"LSTM feature dim should be 1 (HI only), got {X_lstm.shape[2]}"
    )
    assert Y_lstm.shape == (X_lstm.shape[0], 1), (
        f"RUL target shape mismatch: expected ({X_lstm.shape[0]}, 1), got {Y_lstm.shape}"
    )


# ─── Test 3: Sensor count matches config ──────────────────────────────────────

def test_sensor_count():
    """
    Sanity check: CMAPSS_SENSORS has exactly 14 entries (literature standard).
    """
    assert len(CMAPSS_SENSORS) == 14, (
        f"Expected 14 literature-standard CMAPSS sensors, "
        f"got {len(CMAPSS_SENSORS)}: {CMAPSS_SENSORS}"
    )

    # Also confirm no fabricated sensor names are present
    fabricated = {"Vibration", "Efficiency"}
    overlap = fabricated.intersection(set(CMAPSS_SENSORS))
    assert len(overlap) == 0, (
        f"Fabricated sensor(s) found in CMAPSS_SENSORS: {overlap}. "
        "These cause data leakage and must be removed."
    )


# ─── Test 4: CONFIG completeness ──────────────────────────────────────────────

def test_config_keys():
    """Ensures the centralized CONFIG dict has all required hyperparameter keys."""
    required_keys = [
        "window_size", "ae_hidden_dim", "lstm_hidden_dim",
        "learning_rate", "epochs", "batch_size",
        "validation_split", "early_stopping_patience",
        "seeds", "checkpoints_dir"
    ]
    for k in required_keys:
        assert k in CONFIG, (
            f"Missing required config key '{k}' in aiml_core/config.py"
        )

    assert isinstance(CONFIG["seeds"], list) and len(CONFIG["seeds"]) >= 1, \
        "CONFIG['seeds'] must be a non-empty list."
    assert CONFIG["validation_split"] > 0.0 and CONFIG["validation_split"] < 1.0, \
        "CONFIG['validation_split'] must be in (0, 1)."
    assert CONFIG["early_stopping_patience"] >= 1, \
        "CONFIG['early_stopping_patience'] must be at least 1."
