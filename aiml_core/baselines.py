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
from aiml_core.config import CONFIG
from aiml_core.train_utils import split_train_val_engines

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
    X_val: np.ndarray,
    Y_val: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 0.01,
    early_stopping_patience: int = 4,
    seed: int = 42
) -> Tuple[float, float, np.ndarray]:
    """Generic train-and-eval loop with early stopping and best-epoch state recovery."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    X_tr_t = torch.FloatTensor(X_train).to(device)
    Y_tr_t = torch.FloatTensor(Y_train).to(device)
    X_va_t = torch.FloatTensor(X_val).to(device)
    Y_va_t = torch.FloatTensor(Y_val).to(device)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(X_tr_t.size(0))
        for i in range(0, X_tr_t.size(0), batch_size):
            idx = perm[i : i + batch_size]
            bx, by = X_tr_t[idx], Y_tr_t[idx]
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()

        # Validation eval
        model.eval()
        with torch.no_grad():
            val_pred = model(X_va_t)
            val_loss = float(criterion(val_pred, Y_va_t).item())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    X_te = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        pred_test = model(X_te).cpu().numpy().flatten()

    rmse = float(np.sqrt(np.mean((pred_test - Y_test.flatten()) ** 2)))
    score = compute_nasa_score(Y_test.flatten(), pred_test)
    return rmse, score, pred_test


def run_baseline_suite(
    dm: DatasetManager,
    seeds: List[int] = None,
    epochs: int = None,
    window_size: int = 30
) -> Dict[str, Dict]:
    """
    Trains PlainLSTM and CNN-LSTM on FD001 and evaluates on FD001 (same-domain).
    Runs across multiple random seeds and returns mean ± std metrics.
    Reuses training configs and split ratios, and outputs seed 42 predictions.
    """
    if seeds is None:
        seeds = CONFIG["seeds"]
    if epochs is None:
        epochs = CONFIG["epochs"]

    print("[baselines] Loading FD001 for baseline training...")
    train_df, test_df = dm.get_dataset("FD001")
    train_df = train_df.ffill().bfill()
    test_df = test_df.ffill().bfill()

    sensor_list = CMAPSS_SENSORS
    results = {}

    # Initialize results structures
    for model_name, ModelClass in [("PlainLSTM", PlainLSTM), ("CNN-LSTM", CNNLSTMModel)]:
        print(f"[baselines] Running {model_name} across {len(seeds)} seeds...")
        rmses, scores = [], []
        pred_test_seed_42 = None
        for seed in seeds:
            # 1. Split training engines at engine level to avoid leakage
            train_sub, val_sub = split_train_val_engines(train_df, CONFIG["validation_split"], seed)

            # 2. Per-regime normalization
            norm_train = normalize_regimes(
                train_sub, sensor_list,
                regime_col="regime" if "regime" in train_sub.columns else "Setting1"
            )
            norm_val = normalize_regimes(
                val_sub, sensor_list,
                regime_col="regime" if "regime" in val_sub.columns else "Setting1"
            )
            norm_test = normalize_regimes(
                test_df, sensor_list,
                regime_col="regime" if "regime" in test_df.columns else "Setting1"
            )

            # 3. Fit global z-score parameters solely on train subset
            raw_train = norm_train[sensor_list].values
            mean_s = raw_train.mean(axis=0)
            std_s = raw_train.std(axis=0)
            std_s[std_s == 0] = 1.0

            norm_train[sensor_list] = (norm_train[sensor_list].values - mean_s) / std_s
            norm_val[sensor_list] = (norm_val[sensor_list].values - mean_s) / std_s
            norm_test[sensor_list] = (norm_test[sensor_list].values - mean_s) / std_s

            # 4. Prepare sliding windows
            X_train, Y_train = _prepare_raw_sensor_windows(norm_train, sensor_list, window_size)
            X_val, Y_val = _prepare_raw_sensor_windows(norm_val, sensor_list, window_size)
            X_test, Y_test = _prepare_raw_sensor_windows(norm_test, sensor_list, window_size)

            if len(X_train) == 0 or len(X_test) == 0:
                print("[baselines] Insufficient data — returning dummy results.")
                dummy = {"rmse_mean": 35.0, "rmse_std": 0.0, "score_mean": 9999.0, "score_std": 0.0}
                return {"PlainLSTM": dummy, "CNN-LSTM": dummy}

            if len(X_val) == 0:
                X_val, Y_val = X_train.copy(), Y_train.copy()

            n_sensors = X_train.shape[-1]
            torch.manual_seed(seed)
            if ModelClass == PlainLSTM:
                model = PlainLSTM(input_dim=n_sensors, hidden_dim=32).to(device)
            else:
                model = CNNLSTMModel(input_dim=n_sensors, hidden_dim=32).to(device)

            rmse, score, pred_test = _train_and_eval_model(
                model, X_train, Y_train, X_val, Y_val, X_test, Y_test,
                epochs=epochs, batch_size=CONFIG["batch_size"],
                learning_rate=CONFIG["learning_rate"],
                early_stopping_patience=CONFIG["early_stopping_patience"],
                seed=seed
            )
            rmses.append(rmse)
            scores.append(score)
            if seed == 42:
                pred_test_seed_42 = pred_test
            print(f"  seed={seed}  RMSE={rmse:.2f}  NASA={score:.1f}")

        results[model_name] = {
            "rmse_mean": round(float(np.mean(rmses)), 2),
            "rmse_std": round(float(np.std(rmses)), 2),
            "score_mean": round(float(np.mean(scores)), 1),
            "score_std": round(float(np.std(scores)), 1),
            "rmse_str": f"{np.mean(rmses):.2f} ± {np.std(rmses):.2f}",
            "score_str": f"{np.mean(scores):.1f} ± {np.std(scores):.1f}",
            "per_seed": [{"seed": s, "rmse": r, "score": sc} for s, r, sc in zip(seeds, rmses, scores)],
            "predictions_seed_42": pred_test_seed_42
        }

    return results
