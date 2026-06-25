"""
faithfulness.py — PMA Explainer Faithfulness Validation

Implements deletion-insertion tests to empirically verify that PMA attributions
are faithful to the model. A faithful explainer should cause faster prediction
degradation when high-attribution features are zeroed first.

Methods compared:
  1. PMA-guided deletion (our method)
  2. Integrated Gradients deletion (strong physical baseline)
  3. Gradient × Input guided deletion (reference baseline)
  4. Random deletion (lower bound — any faithful method should beat this)

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
    Computes d(RUL_pred)/d(x) * x for each feature dimension, then
    averages over the time axis to get a (n_sensors,) attribution vector.
    """
    ae_criterion = nn.MSELoss()
    err_offset = mean_recon_err * 0.95

    x_t = torch.FloatTensor(x_norm_3d).to(device).requires_grad_(True)

    ae_model.eval()
    lstm_model.eval()

    recon = ae_model(x_t)
    errs = torch.mean((recon - x_t) ** 2, dim=2).squeeze(0)  # shape: (30,)

    hi_vals = []
    for i in range(errs.shape[0]):
        e = errs[i]
        hi_vals.append(100.0 * torch.exp(-torch.clamp(e - err_offset, min=0.0) / ae_threshold))
    hi_t = torch.stack(hi_vals).unsqueeze(0).unsqueeze(2)  # (1, 30, 1)

    rul_pred = lstm_model(hi_t, mc_dropout=False)
    scalar = rul_pred.squeeze()

    scalar.backward()

    grad = x_t.grad.detach().cpu().numpy()  # (1, 30, n_sensors)
    x_np = x_norm_3d  # (1, 30, n_sensors)

    grad_x_input = grad * x_np  # element-wise
    attributions = np.mean(np.abs(grad_x_input[0]), axis=0)  # (n_sensors,)

    return attributions


def compute_integrated_gradients_attributions(
    ae_model: nn.Module,
    lstm_model: nn.Module,
    x_norm_3d: np.ndarray,
    ae_threshold: float,
    mean_recon_err: float,
    steps: int = 25
) -> np.ndarray:
    """
    Integrated Gradients (IG) baseline.
    Interpolates linearly from a zero baseline to x_norm_3d and integrates path gradients.
    """
    ae_criterion = nn.MSELoss()
    err_offset = mean_recon_err * 0.95
    n_sensors = x_norm_3d.shape[-1]

    baseline = np.zeros_like(x_norm_3d)
    accumulated_grads = np.zeros_like(x_norm_3d)

    ae_model.eval()
    lstm_model.eval()

    for step in range(steps + 1):
        alpha = step / steps
        interpolated = baseline + alpha * (x_norm_3d - baseline)
        interpolated_t = torch.FloatTensor(interpolated).to(device).requires_grad_(True)

        recon = ae_model(interpolated_t)
        errs = torch.mean((recon - interpolated_t) ** 2, dim=2).squeeze(0)  # (30,)

        hi_vals = []
        for i in range(errs.shape[0]):
            e = errs[i]
            hi_vals.append(100.0 * torch.exp(-torch.clamp(e - err_offset, min=0.0) / ae_threshold))
        hi_t = torch.stack(hi_vals).unsqueeze(0).unsqueeze(2)  # (1, 30, 1)

        rul_pred = lstm_model(hi_t, mc_dropout=False)
        scalar = rul_pred.squeeze()

        # Zero gradients before backward pass
        ae_model.zero_grad()
        lstm_model.zero_grad()
        scalar.backward()

        grad = interpolated_t.grad.detach().cpu().numpy()
        accumulated_grads += grad

    avg_grads = accumulated_grads / (steps + 1)
    integrated_grads = avg_grads * (x_norm_3d - baseline)
    attributions = np.mean(np.abs(integrated_grads[0]), axis=0)  # (n_sensors,)

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


def compute_audc(curve: np.ndarray) -> float:
    """Area Under the Deletion Curve (normalized to [0, 1])."""
    n = len(curve)
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
    n_samples: int = 100, # Increased sample size
    output_path: str = "webdev_core/static/faithfulness_plot.png"
) -> Dict[str, Any]:
    """
    Main entry point for faithfulness validation.
    """
    all_engine_ids = norm_test_df["engine_id"].unique()
    rng = np.random.default_rng(99)

    windows = []
    for eid in all_engine_ids:
        eng_df = norm_test_df[norm_test_df["engine_id"] == eid].sort_values("cycle")
        sensor_vals = eng_df[sensor_list].values
        n = len(sensor_vals)
        if n < 30:
            continue
        # Sample windows evenly across the trajectory
        indices = np.linspace(0, n - 30, num=max(2, n_samples // len(all_engine_ids) + 1), dtype=int)
        for start in indices:
            window = sensor_vals[start:start + 30]
            windows.append(window)
            if len(windows) >= n_samples:
                break
        if len(windows) >= n_samples:
            break

    if len(windows) == 0:
        print("[faithfulness] No windows to evaluate — skipping.")
        return {
            "pma_audc": 0.0, "pma_ci": 0.0,
            "ig_audc": 0.0, "ig_ci": 0.0,
            "gradient_audc": 0.0, "gradient_ci": 0.0,
            "random_audc": 0.0, "random_ci": 0.0
        }

    windows = np.array(windows[:n_samples])
    N, seq_len, n_sensors = windows.shape
    print(f"[faithfulness] Evaluating deletion curves on {N} test windows...")

    baseline_3d = np.zeros((1, seq_len, n_sensors))
    def rul_scorer(x_3d):
        return _sensor_to_rul(ae_model, lstm_model, x_3d, ae_threshold, mean_recon_err)

    pma_explainer = PMAExplainer(rul_scorer, baseline_3d)

    pma_audcs = []
    ig_audcs = []
    gi_audcs = []
    rand_audcs = []

    pma_curves = []
    ig_curves = []
    gi_curves = []
    rand_curves = []

    for i in range(N):
        x = windows[i:i+1]
        
        # Calculate attributions for each explainer
        try:
            pma_attr = pma_explainer.explain(x)
        except Exception:
            pma_attr = np.zeros(n_sensors)
            
        try:
            ig_attr = compute_integrated_gradients_attributions(
                ae_model, lstm_model, x, ae_threshold, mean_recon_err
            )
        except Exception:
            ig_attr = np.zeros(n_sensors)
            
        try:
            gi_attr = compute_gradient_x_input_attributions(
                ae_model, lstm_model, x, ae_threshold, mean_recon_err
            )
        except Exception:
            gi_attr = np.zeros(n_sensors)

        pma_order = np.argsort(-np.abs(pma_attr))
        ig_order = np.argsort(-np.abs(ig_attr))
        gi_order = np.argsort(-np.abs(gi_attr))
        rand_order = rng.permutation(n_sensors)

        c_pma = _run_deletion_curve(ae_model, lstm_model, x, ae_threshold, mean_recon_err, pma_order.tolist())
        c_ig = _run_deletion_curve(ae_model, lstm_model, x, ae_threshold, mean_recon_err, ig_order.tolist())
        c_gi = _run_deletion_curve(ae_model, lstm_model, x, ae_threshold, mean_recon_err, gi_order.tolist())
        c_rand = _run_deletion_curve(ae_model, lstm_model, x, ae_threshold, mean_recon_err, rand_order.tolist())

        pma_curves.append(c_pma)
        ig_curves.append(c_ig)
        gi_curves.append(c_gi)
        rand_curves.append(c_rand)

        pma_audcs.append(compute_audc(c_pma))
        ig_audcs.append(compute_audc(c_ig))
        gi_audcs.append(compute_audc(c_gi))
        rand_audcs.append(compute_audc(c_rand))

    # Mean curves
    pma_mean = np.mean(pma_curves, axis=0)
    ig_mean = np.mean(ig_curves, axis=0)
    gi_mean = np.mean(gi_curves, axis=0)
    rand_mean = np.mean(rand_curves, axis=0)

    # Compute stats and 95% Confidence Intervals (CI = 1.96 * std / sqrt(N))
    def get_stats(audcs):
        mean = float(np.mean(audcs))
        std = float(np.std(audcs))
        ci = 1.96 * std / np.sqrt(N)
        return round(mean, 4), round(ci, 4)

    pma_m, pma_ci = get_stats(pma_audcs)
    ig_m, ig_ci = get_stats(ig_audcs)
    gi_m, gi_ci = get_stats(gi_audcs)
    rand_m, rand_ci = get_stats(rand_audcs)

    print(f"[faithfulness] PMA AUDC: {pma_m} ± {pma_ci}")
    print(f"[faithfulness] Integrated Gradients AUDC: {ig_m} ± {ig_ci}")
    print(f"[faithfulness] Gradient×Input AUDC: {gi_m} ± {gi_ci}")
    print(f"[faithfulness] Random AUDC: {rand_m} ± {rand_ci}")

    # Generate faithfulness deletion curves plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x_axis = np.linspace(0, 100, len(pma_mean))

        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.set_facecolor("#0d1322")
        fig.patch.set_facecolor("#060a13")

        # Colorblind safe color palette: Cyan, Orange, Blue, Grey
        ax.plot(x_axis, rand_mean, color="#8397b5", linewidth=1.5, linestyle="--", label=f"Random (AUDC={rand_m:.3f}±{rand_ci:.3f})")
        ax.plot(x_axis, gi_mean, color="#e66101", linewidth=1.8, linestyle="-.", label=f"Gradient×Input (AUDC={gi_m:.3f}±{gi_ci:.3f})")
        ax.plot(x_axis, ig_mean, color="#5e3c99", linewidth=2.0, linestyle=":", label=f"Integrated Gradients (AUDC={ig_m:.3f}±{ig_ci:.3f})")
        ax.plot(x_axis, pma_mean, color="#00f0ff", linewidth=2.5, label=f"PMA — Ours (AUDC={pma_m:.3f}±{pma_ci:.3f})")

        ax.set_xlabel("Features Deleted (%)", color="#8397b5", fontsize=9)
        ax.set_ylabel("Mean Predicted RUL (cycles)", color="#8397b5", fontsize=9)
        ax.set_title(f"Faithfulness Deletion Curves (N={N} windows, 95% Confidence Intervals)", color="white", fontsize=10)
        ax.tick_params(colors="#8397b5", labelsize=8)
        ax.spines[:].set_color("#4d607c")
        ax.grid(color="#4d607c", alpha=0.15, linestyle="--")
        ax.legend(facecolor="#0d1322", edgecolor="#4d607c", labelcolor="white", fontsize=8)

        note_text = "Lower AUDC reflects faster prediction drop when deleting top features (higher faithfulness)."
        fig.text(0.5, -0.04, note_text, ha="center", color="#546682", fontsize=7, style="italic")

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, facecolor="#060a13", bbox_inches="tight")
        plt.close()
        print(f"[faithfulness] Faithfulness deletion curve saved to {output_path}")
    except Exception as e:
        print(f"[faithfulness] Could not generate plot: {e}")

    return {
        "pma_audc": pma_m, "pma_ci": pma_ci,
        "ig_audc": ig_m, "ig_ci": ig_ci,
        "gradient_audc": gi_m, "gradient_ci": gi_ci,
        "random_audc": rand_m, "random_ci": rand_ci
    }
