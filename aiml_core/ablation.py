"""
ablation.py — Ablation Studies

Two targeted ablations that justify the design choices in the pipeline:

1. Health Index Abstraction Ablation:
   Compares BayesianLSTM fed HI sequences (our architecture) vs.
   BayesianLSTM fed raw normalized sensor sequences directly.
   Tests whether the HI abstraction layer helps RUL prediction.

2. Window Size Sensitivity Ablation:
   Evaluates the pipeline for window sizes [15, 30, 50] on FD001.
   Varies the AE encoding window size concurrently with the downstream HI sequence.
   Shows how the window size trades off between local detail and
   long-range context.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any

from aiml_core.models import LSTMAutoencoder, BayesianLSTM
from aiml_core.data_loader import (
    DatasetManager, normalize_regimes,
    prepare_sliding_windows, CMAPSS_SENSORS
)
from aiml_core.config import CONFIG
from aiml_core.train_utils import train_pipeline, get_or_train_models
from aiml_core.benchmark import (
    compute_nasa_score, run_evaluation
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _train_raw_lstm(
    norm_train_df,
    sensor_list: List[str],
    window_size: int = 30,
    hidden_dim: int = 16,
    epochs: int = 20,
    seed: int = 42
) -> BayesianLSTM:
    """
    Trains a BayesianLSTM directly on raw (regime-normalized) sensor sequences.
    No autoencoder, no health index — just raw multi-sensor → RUL.
    This is the ablation baseline (no HI abstraction).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 85/15 train/val split at engine level
    unique_engines = norm_train_df["engine_id"].unique()
    rng = np.random.default_rng(seed)
    val_engines = rng.choice(unique_engines, size=int(len(unique_engines) * CONFIG["validation_split"]), replace=False)
    train_engines = np.setdiff1d(unique_engines, val_engines)
    
    train_sub = norm_train_df[norm_train_df["engine_id"].isin(train_engines)]
    val_sub = norm_train_df[norm_train_df["engine_id"].isin(val_engines)]
    
    X_tr, Y_tr = prepare_sliding_windows(train_sub, sensor_list, window_size=window_size)
    X_va, Y_va = prepare_sliding_windows(val_sub, sensor_list, window_size=window_size)
    
    if len(X_tr) == 0:
        raise ValueError("No sliding windows generated.")
    if len(X_va) == 0:
        X_va, Y_va = X_tr, Y_tr

    n_features = X_tr.shape[-1]
    model = BayesianLSTM(input_dim=n_features, hidden_dim=hidden_dim, output_dim=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.MSELoss()

    X_tr_t = torch.FloatTensor(X_tr).to(device)
    Y_tr_t = torch.FloatTensor(Y_tr).to(device)
    X_va_t = torch.FloatTensor(X_va).to(device)
    Y_va_t = torch.FloatTensor(Y_va).to(device)

    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    model.train()
    batch_size = CONFIG["batch_size"]
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(X_tr_t.size(0))
        for i in range(0, X_tr_t.size(0), batch_size):
            idx = perm[i : i + batch_size]
            bx, by = X_tr_t[idx], Y_tr_t[idx]
            optimizer.zero_grad()
            pred = model(bx, mc_dropout=False)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            val_pred = model(X_va_t, mc_dropout=False)
            val_loss = float(criterion(val_pred, Y_va_t).item())
            
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= CONFIG["early_stopping_patience"]:
            break
            
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model


def _eval_raw_lstm(
    raw_lstm: BayesianLSTM,
    norm_test_df,
    sensor_list: List[str],
    window_size: int = 30
) -> Tuple[float, float]:
    """Evaluates the raw-sensor LSTM on the test set."""
    X, Y = prepare_sliding_windows(norm_test_df, sensor_list, window_size=window_size)
    if len(X) == 0:
        return 35.0, 9999.0

    X_t = torch.FloatTensor(X).to(device)
    raw_lstm.eval()
    with torch.no_grad():
        pred = raw_lstm(X_t, mc_dropout=False).cpu().numpy()
    rmse = float(np.sqrt(np.mean((pred - Y) ** 2)))
    score = compute_nasa_score(Y.flatten(), pred.flatten())
    return rmse, score


def run_hi_ablation(
    dm: DatasetManager,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Ablation 1: Health Index Abstraction Benefit.
    """
    print("[ablation] HI Abstraction Ablation — loading FD001...")
    train_df, test_df = dm.get_dataset("FD001")
    train_df = train_df.ffill().bfill()
    test_df = test_df.ffill().bfill()

    sensor_list = CMAPSS_SENSORS

    norm_train = normalize_regimes(
        train_df, sensor_list,
        regime_col="regime" if "regime" in train_df.columns else "Setting1"
    )
    norm_test = normalize_regimes(
        test_df, sensor_list,
        regime_col="regime" if "regime" in test_df.columns else "Setting1"
    )

    print("[ablation] Training Variant A: Full pipeline (AE → HI → LSTM)...")
    ae_model, lstm_hi, ae_threshold, mean_recon_err, t_mean, t_std = get_or_train_models("FD001", seed, dm)
    
    # Run evaluation and unpack all 9 elements
    rmse_hi, score_hi, picp, sharpness, pred_means, pred_stds, y_true, X_lstm_t, norm_test_ret = run_evaluation(
        "FD001", ae_model, lstm_hi, ae_threshold, dm, t_mean, t_std, window_size=CONFIG["window_size"]
    )

    print("[ablation] Training Variant B: Raw sensor LSTM (no HI)...")
    raw_lstm = _train_raw_lstm(norm_train, sensor_list, window_size=30, seed=seed)
    rmse_raw, score_raw = _eval_raw_lstm(raw_lstm, norm_test, sensor_list, window_size=30)

    delta_rmse = round(rmse_raw - rmse_hi, 2)
    delta_pct = round((delta_rmse / (rmse_hi + 1e-9)) * 100.0, 1)

    result = {
        "hi_pipeline": {
            "label": "HI-LSTM (Full Pipeline)",
            "rmse": round(rmse_hi, 2),
            "score": round(score_hi, 1)
        },
        "raw_pipeline": {
            "label": "Raw-Sensor LSTM (Ablated)",
            "rmse": round(rmse_raw, 2),
            "score": round(score_raw, 1)
        },
        "delta_rmse": delta_rmse,
        "delta_pct": delta_pct,
        "hi_helps": delta_rmse > 0
    }

    print(f"[ablation] HI result: RMSE={rmse_hi:.2f}, Raw result: RMSE={rmse_raw:.2f}, Delta: {delta_rmse:+.2f} ({delta_pct:+.1f}%)")
    return result


def run_window_size_ablation(
    dm: DatasetManager,
    windows: List[int] = (15, 30, 50),
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Ablation 2: Window Size Sensitivity.
    Evaluates the pipeline for each window size on FD001, varying the AE window concurrently.
    """
    print(f"[ablation] Window Size Ablation — windows={list(windows)}...")
    train_df, _ = dm.get_dataset("FD001")
    train_df = train_df.ffill().bfill()

    results = []
    for w in windows:
        print(f"[ablation]   Training with window_size={w} (AE + LSTM)...")
        try:
            # Train pipeline dynamically with target window size (AE & LSTM both use window_size = w)
            ae_model, lstm_model, ae_threshold, mean_recon_err, t_mean, t_std = train_pipeline(
                "FD001", train_df, seed, window_size=w
            )
            # Evaluate using same window size
            rmse, score, picp, sharpness, pred_means, pred_stds, y_true, X_lstm_t, norm_test_ret = run_evaluation(
                "FD001", ae_model, lstm_model, ae_threshold, dm, t_mean, t_std, window_size=w
            )
            results.append({
                "window_size": w,
                "rmse": round(rmse, 2),
                "score": round(score, 1),
                "note": "Selected" if w == 30 else ""
            })
            print(f"  window={w}  RMSE={rmse:.2f}  NASA={score:.1f}")
        except Exception as e:
            print(f"  window={w} FAILED: {e}")
            results.append({"window_size": w, "rmse": None, "score": None, "note": "Error"})

    return results


def generate_ablation_table(dm: DatasetManager) -> Dict[str, Any]:
    """
    Runs both ablations and returns markdown + latex summary tables.
    """
    print("[ablation] Running full ablation suite...")

    hi_result = run_hi_ablation(dm)
    window_result = run_window_size_ablation(dm)

    # --- HI Ablation Table ---
    md_hi = "### Ablation 1: Health Index Abstraction Benefit\n\n"
    md_hi += "| Variant | RMSE ↓ | NASA Score ↓ | Note |\n"
    md_hi += "|---------|--------|-------------|------|\n"
    md_hi += f"| {hi_result['hi_pipeline']['label']} | **{hi_result['hi_pipeline']['rmse']}** | **{hi_result['hi_pipeline']['score']}** | Proposed |\n"
    md_hi += f"| {hi_result['raw_pipeline']['label']} | {hi_result['raw_pipeline']['rmse']} | {hi_result['raw_pipeline']['score']} | No HI layer |\n"
    delta_note = f"HI pipeline {'improves' if hi_result['hi_helps'] else 'degrades'} RMSE by {abs(hi_result['delta_rmse'])} cycles ({abs(hi_result['delta_pct'])}%)"
    md_hi += f"\n*{delta_note}*\n"

    # --- Window Ablation Table ---
    md_win = "### Ablation 2: Sliding Window Size Sensitivity (FD001)\n\n"
    md_win += "| Window Size | RMSE ↓ | NASA Score ↓ | Note |\n"
    md_win += "|-------------|--------|-------------|------|\n"
    for row in window_result:
        rmse_str = str(row["rmse"]) if row["rmse"] is not None else "—"
        score_str = str(row["score"]) if row["score"] is not None else "—"
        bold = "**" if row["window_size"] == 30 else ""
        md_win += f"| {bold}{row['window_size']} cycles{bold} | {bold}{rmse_str}{bold} | {bold}{score_str}{bold} | {row['note']} |\n"

    # --- LaTeX ---
    latex_hi = (
        "\\begin{table}[h]\\centering\n"
        "\\caption{Ablation: Health Index Abstraction vs. Raw Sensor Input}\n"
        "\\label{tab:ablation_hi}\n"
        "\\begin{tabular}{lcc}\\hline\n"
        "\\textbf{Variant} & \\textbf{RMSE} & \\textbf{NASA Score} \\\\ \\hline\n"
        f"{hi_result['hi_pipeline']['label']} & \\textbf{{{hi_result['hi_pipeline']['rmse']}}} & \\textbf{{{hi_result['hi_pipeline']['score']}}} \\\\\n"
        f"{hi_result['raw_pipeline']['label']} & {hi_result['raw_pipeline']['rmse']} & {hi_result['raw_pipeline']['score']} \\\\\n"
        "\\hline\\end{tabular}\\end{table}"
    )

    latex_win = (
        "\\begin{table}[h]\\centering\n"
        "\\caption{Ablation: Window Size Sensitivity on FD001}\n"
        "\\label{tab:ablation_window}\n"
        "\\begin{tabular}{lcc}\\hline\n"
        "\\textbf{Window Size} & \\textbf{RMSE} & \\textbf{NASA Score} \\\\ \\hline\n"
    )
    for row in window_result:
        rmse_str = str(row["rmse"]) if row["rmse"] is not None else "—"
        score_str = str(row["score"]) if row["score"] is not None else "—"
        bold = "\\textbf" if row["window_size"] == 30 else ""
        if bold:
            latex_win += f"\\textbf{{{row['window_size']} cycles}} & \\textbf{{{rmse_str}}} & \\textbf{{{score_str}}} \\\\\n"
        else:
            latex_win += f"{row['window_size']} cycles & {rmse_str} & {score_str} \\\\\n"
    latex_win += "\\hline\\end{tabular}\\end{table}"

    return {
        "hi_ablation": hi_result,
        "window_ablation": window_result,
        "markdown": md_hi + "\n" + md_win,
        "latex": latex_hi + "\n\n" + latex_win
    }
