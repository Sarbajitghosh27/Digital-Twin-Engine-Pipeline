"""
baselines.py — Multi-seed Baseline Evaluation Suite

Trains and evaluates PlainLSTM and CNN-LSTM on raw sensor windows
(no Health Index abstraction) and reports RMSE and NASA Score.

Multi-seed runs (3 seeds by default) with mean ± std reporting provide
statistical weight to all benchmark comparisons.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict

from aiml_core.models import PlainLSTM, CNNLSTMModel
from aiml_core.data_loader import (
    DatasetManager, normalize_regimes, CMAPSS_SENSORS
)
from aiml_core.benchmark import compute_nasa_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prepare_raw_sensor_windows(
    df,
    sensor_list: List[str],
    window_size: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepares sliding windows of raw (regime-normalized) sensor data
    for direct use as input to PlainLSTM / CNN-LSTM (no HI stage).

    Returns:
        X: (N, window_size, n_sensors)
        Y: (N, 1)
    """
    X_list, Y_list = [], []
    for eid in df["engine_id"].unique():
        eng = df[df["engine_id"] == eid].sort_values("cycle")
        if "RUL_actual" not in eng.columns:
            continue
        vals = eng[sensor_list].values
        ruls = eng["RUL_actual"].values
        n = len(eng)
        if n < window_size:
            continue
        for i in range(n - window_size + 1):
            X_list.append(vals[i : i + window_size])
            Y_list.append(ruls[i + window_size - 1])

    if len(X_list) == 0:
        return np.zeros((0, window_size, len(sensor_list))), np.zeros((0, 1))
    return np.array(X_list), np.array(Y_list).reshape(-1, 1)


def _train_and_eval_model(
    model: nn.Module,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    epochs: int = 5,
    batch_size: int = 64,
    seed: int = 42
) -> Tuple[float, float]:
    """Generic train-and-eval loop for any model that accepts (x) → scalar."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    X_t = torch.FloatTensor(X_train).to(device)
    Y_t = torch.FloatTensor(Y_train).to(device)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(X_t.size(0))
        for i in range(0, X_t.size(0), batch_size):
            idx = perm[i : i + batch_size]
            bx, by = X_t[idx], Y_t[idx]
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()

    model.eval()
    X_te = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        pred_test = model(X_te).cpu().numpy()

    rmse = float(np.sqrt(np.mean((pred_test - Y_test) ** 2)))
    score = compute_nasa_score(Y_test.flatten(), pred_test.flatten())
    return rmse, score


def run_baseline_suite(
    dm: DatasetManager,
    seeds: List[int] = (42, 123, 7),
    epochs: int = 5,
    window_size: int = 30
) -> Dict[str, Dict]:
    """
    Trains PlainLSTM and CNN-LSTM on FD001 and evaluates on FD001 (same-domain).
    Runs across multiple random seeds and returns mean ± std metrics.

    Returns dict:
    {
        "PlainLSTM":  {"rmse_mean": float, "rmse_std": float, "score_mean": float, "score_std": float},
        "CNN-LSTM":   {...},
    }
    """
    print("[baselines] Loading FD001 for baseline training...")
    train_df, test_df = dm.get_dataset("FD001")
    train_df = train_df.ffill().bfill()
    test_df = test_df.ffill().bfill()

    sensor_list = CMAPSS_SENSORS

    # Per-regime normalization
    norm_train = normalize_regimes(
        train_df, sensor_list,
        regime_col="regime" if "regime" in train_df.columns else "Setting1"
    )
    norm_test = normalize_regimes(
        test_df, sensor_list,
        regime_col="regime" if "regime" in test_df.columns else "Setting1"
    )

    # Global z-score on top of regime normalization
    raw_train = norm_train[sensor_list].values
    mean_s = raw_train.mean(axis=0)
    std_s = raw_train.std(axis=0)
    std_s[std_s == 0] = 1.0
    norm_train[sensor_list] = (norm_train[sensor_list].values - mean_s) / std_s
    norm_test[sensor_list] = (norm_test[sensor_list].values - mean_s) / std_s

    X_train, Y_train = _prepare_raw_sensor_windows(norm_train, sensor_list, window_size)
    X_test, Y_test = _prepare_raw_sensor_windows(norm_test, sensor_list, window_size)

    if len(X_train) == 0 or len(X_test) == 0:
        print("[baselines] Insufficient data — returning dummy results.")
        dummy = {"rmse_mean": 35.0, "rmse_std": 0.0, "score_mean": 9999.0, "score_std": 0.0}
        return {"PlainLSTM": dummy, "CNN-LSTM": dummy}

    n_sensors = X_train.shape[-1]
    results = {}

    for model_name, ModelClass in [("PlainLSTM", PlainLSTM), ("CNN-LSTM", CNNLSTMModel)]:
        print(f"[baselines] Running {model_name} across {len(seeds)} seeds...")
        rmses, scores = [], []
        for seed in seeds:
            torch.manual_seed(seed)
            if ModelClass == PlainLSTM:
                model = PlainLSTM(input_dim=n_sensors, hidden_dim=32).to(device)
            else:
                model = CNNLSTMModel(input_dim=n_sensors, hidden_dim=32).to(device)

            rmse, score = _train_and_eval_model(
                model, X_train, Y_train, X_test, Y_test,
                epochs=epochs, seed=seed
            )
            rmses.append(rmse)
            scores.append(score)
            print(f"  seed={seed}  RMSE={rmse:.2f}  NASA={score:.1f}")

        results[model_name] = {
            "rmse_mean": round(float(np.mean(rmses)), 2),
            "rmse_std": round(float(np.std(rmses)), 2),
            "score_mean": round(float(np.mean(scores)), 1),
            "score_std": round(float(np.std(scores)), 1),
            "rmse_str": f"{np.mean(rmses):.2f} ± {np.std(rmses):.2f}",
            "score_str": f"{np.mean(scores):.1f} ± {np.std(scores):.1f}",
            "per_seed": [{"seed": s, "rmse": r, "score": sc} for s, r, sc in zip(seeds, rmses, scores)]
        }

    return results
