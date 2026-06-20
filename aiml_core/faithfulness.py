"""
faithfulness.py — PMA Explainer Faithfulness Validation

Implements deletion-insertion tests to empirically verify that PMA attributions
are faithful to the model. A faithful explainer should cause faster prediction
degradation when high-attribution features are zeroed first.

Methods compared:
  1. PMA-guided deletion (our method)
  2. Gradient × Input guided deletion (reference baseline)
  3. Random deletion (lower bound — any faithful method should beat this)

The Area Under the Deletion Curve (AUDC) is the primary metric.
Lower AUDC = model degrades faster when removing top features = better faithfulness.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from typing import Callable, List, Tuple, Dict, Any
from aiml_core.explainers import PMAExplainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sensor_to_rul(
    ae_model: nn.Module,
    lstm_model: nn.Module,
    x_norm_3d: np.ndarray,
    ae_threshold: float,
    mean_recon_err: float
) -> float:
    """
    Maps a normalized sensor window (shape: 1 × 30 × n_sensors) to predicted RUL.
    Runs the full HI pipeline: AE reconstruction → HI sequence → LSTM RUL prediction.
    """
    ae_criterion = nn.MSELoss()
    x_t = torch.FloatTensor(x_norm_3d).to(device)

    ae_model.eval()
    lstm_model.eval()

    with torch.no_grad():
        recon = ae_model(x_t)
        # Per-timestep reconstruction error (mean over sensor dim)
        errs = torch.mean((recon - x_t) ** 2, dim=2).squeeze(0).cpu().numpy()

    err_offset = mean_recon_err * 0.95
    hi_list = []
    for e in errs:
        hi_list.append(float(100.0 * np.exp(-max(0.0, e - err_offset) / ae_threshold)))

    hi_t = torch.FloatTensor(hi_list).unsqueeze(0).unsqueeze(2).to(device)
    with torch.no_grad():
        rul_pred = lstm_model(hi_t, mc_dropout=False)
    return float(rul_pred.item())


def compute_gradient_x_input_attributions(
    ae_model: nn.Module,
    lstm_model: nn.Module,
    x_norm_3d: np.ndarray,
    ae_threshold: float,
    mean_recon_err: float
) -> np.ndarray:
    """
    Gradient × Input attribution baseline.

    Computes d(RUL_pred)/d(x) * x  for each feature dimension, then
    averages over the time axis to get a (n_sensors,) attribution vector.
    Uses the full AE → HI → LSTM pipeline with autograd.

    Returns:
        attributions: np.ndarray of shape (n_sensors,)
    """
    ae_criterion = nn.MSELoss()
    err_offset = mean_recon_err * 0.95

    x_t = torch.FloatTensor(x_norm_3d).to(device).requires_grad_(True)

    ae_model.eval()
    lstm_model.eval()

    # Forward through AE
    recon = ae_model(x_t)
    errs = torch.mean((recon - x_t) ** 2, dim=2).squeeze(0)  # shape: (30,)

    # Compute HI sequence (differentiable path)
    hi_vals = []
    for i in range(errs.shape[0]):
        e = errs[i]
        hi_vals.append(100.0 * torch.exp(-torch.clamp(e - err_offset, min=0.0) / ae_threshold))
    hi_t = torch.stack(hi_vals).unsqueeze(0).unsqueeze(2)  # (1, 30, 1)

    # Forward through LSTM
    rul_pred = lstm_model(hi_t, mc_dropout=False)
    scalar = rul_pred.squeeze()

    # Backward
    scalar.backward()

    grad = x_t.grad.detach().cpu().numpy()  # (1, 30, n_sensors)
    x_np = x_norm_3d  # (1, 30, n_sensors)

    # Gradient × Input, mean over time axis
    grad_x_input = grad * x_np  # element-wise
    attributions = np.mean(np.abs(grad_x_input[0]), axis=0)  # (n_sensors,)

    return attributions


def _run_deletion_curve(
    ae_model: nn.Module,
    lstm_model: nn.Module,
    x_norm_3d: np.ndarray,
    ae_threshold: float,
    mean_recon_err: float,
    feature_order: List[int]
) -> np.ndarray:
    """
    Runs the deletion test for a single sample.
    Zeros out features in the given order, one at a time, measuring RUL drop.

    Returns:
        curve: np.ndarray of shape (n_sensors + 1,) — RUL prediction at each deletion step.
    """
    n_sensors = x_norm_3d.shape[-1]
    perturbed = x_norm_3d.copy()
    curve = []

    baseline_pred = _sensor_to_rul(ae_model, lstm_model, perturbed, ae_threshold, mean_recon_err)
    curve.append(baseline_pred)

    for feat_idx in feature_order:
        perturbed[0, :, feat_idx] = 0.0  # zero out entire time series of this feature
        pred = _sensor_to_rul(ae_model, lstm_model, perturbed, ae_threshold, mean_recon_err)
        curve.append(pred)

    return np.array(curve)


def compute_deletion_curves(
    ae_model: nn.Module,
    lstm_model: nn.Module,
    test_windows: np.ndarray,
    pma_attributions_batch: np.ndarray,
    grad_attributions_batch: np.ndarray,
    ae_threshold: float,
    mean_recon_err: float
) -> Dict[str, np.ndarray]:
    """
    Computes mean deletion curves for PMA, Gradient×Input, and Random methods
    across a batch of test windows.

    Args:
        test_windows: (N, 30, n_sensors) — normalized sensor windows
        pma_attributions_batch: (N, n_sensors) — PMA attributions per window
        grad_attributions_batch: (N, n_sensors) — gradient×input per window
        ae_threshold: AE reconstruction error scaling factor
        mean_recon_err: baseline reconstruction error

    Returns:
        dict with keys 'pma', 'gradient', 'random' each mapping to
        a (n_sensors + 1,) array representing mean curve values.
    """
    n_samples, seq_len, n_sensors = test_windows.shape
    rng = np.random.default_rng(42)

    pma_curves = []
    grad_curves = []
    rand_curves = []

    for i in range(n_samples):
        x = test_windows[i:i+1]  # (1, 30, n_sensors)

        # PMA order: descending by attribution magnitude
        pma_order = np.argsort(-np.abs(pma_attributions_batch[i]))
        # Gradient order: descending by gradient×input magnitude
        grad_order = np.argsort(-np.abs(grad_attributions_batch[i]))
        # Random order
        rand_order = rng.permutation(n_sensors)

        pma_curves.append(_run_deletion_curve(ae_model, lstm_model, x, ae_threshold, mean_recon_err, pma_order.tolist()))
        grad_curves.append(_run_deletion_curve(ae_model, lstm_model, x, ae_threshold, mean_recon_err, grad_order.tolist()))
        rand_curves.append(_run_deletion_curve(ae_model, lstm_model, x, ae_threshold, mean_recon_err, rand_order.tolist()))

    pma_mean = np.mean(pma_curves, axis=0)
    grad_mean = np.mean(grad_curves, axis=0)
    rand_mean = np.mean(rand_curves, axis=0)

    return {"pma": pma_mean, "gradient": grad_mean, "random": rand_mean}


def compute_audc(curve: np.ndarray) -> float:
    """Area Under the Deletion Curve (normalized to [0, 1])."""
    n = len(curve)
    # Normalize by the initial prediction to make curves comparable across samples
    if abs(curve[0]) < 1e-6:
        return 0.0
    norm_curve = curve / (abs(curve[0]) + 1e-9)
    return float(np.trapz(norm_curve, dx=1.0 / (n - 1)))


def generate_faithfulness_plot(
    ae_model: nn.Module,
    lstm_model: nn.Module,
    norm_test_df,
    sensor_list: List[str],
    ae_threshold: float,
    mean_recon_err: float,
    n_samples: int = 40,
    output_path: str = "webdev_core/static/faithfulness_plot.png"
) -> Dict[str, float]:
    """
    Main entry point for faithfulness validation.

    1. Samples `n_samples` random windows from the test set
    2. Computes PMA and Gradient×Input attributions per window
    3. Runs deletion curves for all three methods
    4. Generates and saves a publication-ready plot
    5. Returns AUDC scores for each method

    Returns:
        dict: {"pma_audc": float, "gradient_audc": float, "random_audc": float}
    """
    import pandas as pd

    # Sample random HI windows from test data
    all_engine_ids = norm_test_df["engine_id"].unique()
    rng = np.random.default_rng(99)

    windows = []
    for eid in all_engine_ids:
        eng_df = norm_test_df[norm_test_df["engine_id"] == eid].sort_values("cycle")
        sensor_vals = eng_df[sensor_list].values
        n = len(sensor_vals)
        if n < 30:
            continue
        for start in range(0, n - 30, max(1, (n - 30) // (n_samples // len(all_engine_ids) + 1))):
            window = sensor_vals[start:start + 30]
            windows.append(window)
            if len(windows) >= n_samples:
                break
        if len(windows) >= n_samples:
            break

    if len(windows) == 0:
        print("[faithfulness] No windows to evaluate — skipping.")
        return {"pma_audc": 0.0, "gradient_audc": 0.0, "random_audc": 0.0}

    windows = np.array(windows[:n_samples])  # (N, 30, n_sensors)
    N, seq_len, n_sensors = windows.shape
    print(f"[faithfulness] Computing attributions on {N} windows...")

    # Baseline state for PMA
    baseline_3d = np.zeros((1, seq_len, n_sensors))

    # Build scorer function for PMA
    def rul_scorer(x_3d):
        return _sensor_to_rul(ae_model, lstm_model, x_3d, ae_threshold, mean_recon_err)

    pma_attrs_batch = []
    grad_attrs_batch = []

    pma_explainer = PMAExplainer(rul_scorer, baseline_3d)

    for i in range(N):
        x = windows[i:i+1]  # (1, 30, n_sensors)
        try:
            pma_attr = pma_explainer.explain(x)
        except Exception as e:
            pma_attr = np.zeros(n_sensors)
        pma_attrs_batch.append(pma_attr)

        try:
            grad_attr = compute_gradient_x_input_attributions(
                ae_model, lstm_model, x, ae_threshold, mean_recon_err
            )
        except Exception as e:
            grad_attr = np.zeros(n_sensors)
        grad_attrs_batch.append(grad_attr)

    pma_attrs_batch = np.array(pma_attrs_batch)
    grad_attrs_batch = np.array(grad_attrs_batch)

    print("[faithfulness] Running deletion curves...")
    curves = compute_deletion_curves(
        ae_model, lstm_model, windows,
        pma_attrs_batch, grad_attrs_batch,
        ae_threshold, mean_recon_err
    )

    pma_audc = compute_audc(curves["pma"])
    grad_audc = compute_audc(curves["gradient"])
    rand_audc = compute_audc(curves["random"])

    print(f"[faithfulness] AUDC — PMA: {pma_audc:.4f}, Gradient×Input: {grad_audc:.4f}, Random: {rand_audc:.4f}")

    # Generate plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x_axis = np.linspace(0, 100, len(curves["pma"]))

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_facecolor("#0d1322")
        fig.patch.set_facecolor("#060a13")

        ax.plot(x_axis, curves["random"], color="#4d607c", linewidth=1.5, linestyle="--", label=f"Random (AUDC={rand_audc:.3f})")
        ax.plot(x_axis, curves["gradient"], color="#ff8c00", linewidth=2.0, linestyle="-.", label=f"Gradient×Input (AUDC={grad_audc:.3f})")
        ax.plot(x_axis, curves["pma"], color="#00f0ff", linewidth=2.5, label=f"PMA — Ours (AUDC={pma_audc:.3f})")

        ax.set_xlabel("Features Deleted (%)", color="#8397b5", fontsize=9)
        ax.set_ylabel("Mean Predicted RUL", color="#8397b5", fontsize=9)
        ax.set_title("PMA Explainer Faithfulness: Deletion Curve Comparison", color="white", fontsize=10)
        ax.tick_params(colors="#8397b5", labelsize=8)
        ax.spines[:].set_color("#4d607c")
        ax.grid(color="#4d607c", alpha=0.15, linestyle="--")
        ax.legend(facecolor="#0d1322", edgecolor="#4d607c", labelcolor="white", fontsize=8)

        note_text = "Lower AUDC = model prediction degrades faster = explainer is more faithful to the model."
        fig.text(0.5, -0.04, note_text, ha="center", color="#546682", fontsize=7, style="italic")

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, facecolor="#060a13", bbox_inches="tight")
        plt.close()
        print(f"[faithfulness] Plot saved to {output_path}")
    except Exception as e:
        print(f"[faithfulness] Could not generate plot: {e}")

    return {
        "pma_audc": round(pma_audc, 4),
        "gradient_audc": round(grad_audc, 4),
        "random_audc": round(rand_audc, 4)
    }
