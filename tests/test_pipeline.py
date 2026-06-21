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

# pyrefly: ignore [missing-import]
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
    assert CONFIG["early_stopping_patience"] >= 1, "CONFIG['early_stopping_patience'] must be at least 1."
# ─── Test 5: Calibration Sanity Check ─────────────────────────────────────────

def test_calibration_sanity(fd001_datasets):
    """
    Validates that empirical coverage (PICP) tracks target nominal confidence levels.
    Nominal targets: 50%, 80%, 95%.
    """
    from aiml_core.train_utils import get_calibration_metrics
    dm, _, _ = fd001_datasets
    
    cal = get_calibration_metrics("FD001", seed=42, dm=dm)
    
    # Assert PICP is monotonic: wider intervals must cover at least as many points
    picp_50 = cal["cl_50"]["picp"]
    picp_80 = cal["cl_80"]["picp"]
    picp_95 = cal["cl_95"]["picp"]
    
    assert 0.0 <= picp_50 <= 1.0
    assert 0.0 <= picp_80 <= 1.0
    assert 0.0 <= picp_95 <= 1.0
    assert picp_50 <= picp_80 <= picp_95, (
        f"PICP should be non-decreasing with confidence level: "
        f"50% level: {picp_50}, 80% level: {picp_80}, 95% level: {picp_95}"
    )

    # Assert sharpness is monotonic: wider intervals must have larger width
    w_50 = cal["cl_50"]["sharpness"]
    w_80 = cal["cl_80"]["sharpness"]
    w_95 = cal["cl_95"]["sharpness"]
    
    assert 0.0 < w_50 <= w_80 <= w_95, (
        f"Sharpness should be strictly positive and non-decreasing: "
        f"50% level: {w_50}, 80% level: {w_80}, 95% level: {w_95}"
    )


# ─── Test 6: Cache Invalidation Test ─────────────────────────────────────────

def test_cache_invalidation(tmp_path):
    """
    Verifies that altering model weight files changes their SHA256 hashes,
    thereby invalidating explainability cache entries.
    """
    import json
    import hashlib
    from webdev_core.backend import get_file_sha256
    
    # 1. Create dummy checkpoint weights files
    ae_file = tmp_path / "ae_dummy.pt"
    lstm_file = tmp_path / "lstm_dummy.pt"
    
    ae_file.write_bytes(b"initial_ae_weights_state_vector_123")
    lstm_file.write_bytes(b"initial_lstm_weights_state_vector_456")
    
    # 2. Get initial hashes
    h_ae1 = get_file_sha256(str(ae_file))
    h_lstm1 = get_file_sha256(str(lstm_file))
    combined_hash1 = f"{h_ae1}_{h_lstm1}"
    
    # 3. Simulate explainability cache entry
    mock_cache = {}
    engine_id = 1
    cycle = 50
    cache_key1 = f"{combined_hash1}_{engine_id}_{cycle}"
    
    mock_explainers = {
        "anomaly_shap": {"T30": 0.5},
        "rul_shap": {"T30": -1.2},
        "top_anomaly_drivers": [{"sensor": "T30", "val": 0.5}]
    }
    mock_cache[cache_key1] = mock_explainers
    
    # Assert cache hit
    assert cache_key1 in mock_cache
    
    # 4. Modify one model weight checkpoint (retraining simulation)
    ae_file.write_bytes(b"retrained_ae_weights_new_state_vector_789")
    
    # 5. Compute new hashes and assert cache miss
    h_ae2 = get_file_sha256(str(ae_file))
    combined_hash2 = f"{h_ae2}_{h_lstm1}"
    cache_key2 = f"{combined_hash2}_{engine_id}_{cycle}"
    
    # Assert that cache invalidation works (mismatched hash misses cache)
    assert h_ae1 != h_ae2, "SHA256 hashes must differ after modifying file bytes."
    assert cache_key2 not in mock_cache, (
        "Retrained model weights hash must cause cache miss to prevent serving stale explainability values."
    )

