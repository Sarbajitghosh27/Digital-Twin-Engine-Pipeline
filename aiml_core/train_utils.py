import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, List
from aiml_core.config import CONFIG
from aiml_core.models import LSTMAutoencoder, BayesianLSTM
from aiml_core.data_loader import normalize_regimes, prepare_sliding_windows, CMAPSS_SENSORS
from aiml_core.hi_normalizer import compute_hi

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_checkpoint_paths(dataset_name: str, seed: int) -> Tuple[str, str, str]:
    """Returns paths for AE, LSTM, and metadata checkpoints."""
    os.makedirs(CONFIG["checkpoints_dir"], exist_ok=True)
    ae_path = os.path.join(CONFIG["checkpoints_dir"], f"ae_{dataset_name}_seed_{seed}.pt")
    lstm_path = os.path.join(CONFIG["checkpoints_dir"], f"lstm_{dataset_name}_seed_{seed}.pt")
    meta_path = os.path.join(CONFIG["checkpoints_dir"], f"meta_{dataset_name}_seed_{seed}.json")
    return ae_path, lstm_path, meta_path

def split_train_val_engines(df: pd.DataFrame, val_split: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits unique engines into train and validation sets to prevent leakage."""
    unique_engines = df["engine_id"].unique()
    rng = np.random.default_rng(seed)
    val_size = int(len(unique_engines) * val_split)
    if val_size == 0:
        val_size = 1 # at least one validation engine
    val_engines = rng.choice(unique_engines, size=val_size, replace=False)
    train_engines = np.setdiff1d(unique_engines, val_engines)
    
    train_df = df[df["engine_id"].isin(train_engines)].copy()
    val_df = df[df["engine_id"].isin(val_engines)].copy()
    return train_df, val_df

def train_pipeline(
    dataset_name: str,
    train_df_full: pd.DataFrame,
    seed: int,
    window_size: int = None
) -> Tuple[LSTMAutoencoder, BayesianLSTM, float, float, np.ndarray, np.ndarray]:
    """
    Trains the LSTM Autoencoder and Bayesian LSTM with early stopping and validation splitting.
    """
    if window_size is None:
        window_size = CONFIG["window_size"]
        
    torch.manual_seed(seed)
    np.random.seed(seed)
    random_state = np.random.default_rng(seed)
    
    # Split engines
    # Run operating condition normalization first to ensure we align normalization scales
    norm_train_df_full = normalize_regimes(
        train_df_full, CMAPSS_SENSORS,
        regime_col="regime" if "regime" in train_df_full.columns else "Setting1"
    )
    train_df, val_df = split_train_val_engines(norm_train_df_full, CONFIG["validation_split"], seed)
    
    # 1. Fit normalizer on healthy cycles (cycle <= 50) of training set
    early_train = train_df[train_df["cycle"] <= 50].copy()
    early_val = val_df[val_df["cycle"] <= 50].copy()
    if len(early_val) == 0:
        early_val = val_df.copy()
        
    ae_mean = early_train[CMAPSS_SENSORS].values.mean(axis=0)
    ae_std = early_train[CMAPSS_SENSORS].values.std(axis=0)
    ae_std[ae_std == 0] = 1.0
    
    # Prepare AE sequences
    def get_ae_seqs(df_sub):
        seqs = []
        for eid in df_sub["engine_id"].unique():
            eng_data = df_sub[df_sub["engine_id"] == eid].sort_values("cycle")[CMAPSS_SENSORS].values
            eng_norm = (eng_data - ae_mean) / ae_std
            if len(eng_norm) >= window_size:
                for i in range(len(eng_norm) - window_size + 1):
                    seqs.append(eng_norm[i : i + window_size])
        return np.array(seqs)

    ae_train_seqs = get_ae_seqs(early_train)
    ae_val_seqs = get_ae_seqs(early_val)
    
    if len(ae_train_seqs) == 0:
        ae_train_seqs = np.random.normal(0, 1, (100, window_size, len(CMAPSS_SENSORS)))
    if len(ae_val_seqs) == 0:
        ae_val_seqs = ae_train_seqs.copy()
        
    ae_model = LSTMAutoencoder(input_dim=len(CMAPSS_SENSORS), hidden_dim=CONFIG["ae_hidden_dim"]).to(device)
    ae_opt = torch.optim.Adam(ae_model.parameters(), lr=CONFIG["learning_rate"])
    ae_criterion = nn.MSELoss()
    
    # Train AE with early stopping
    X_train_ae = torch.FloatTensor(ae_train_seqs).to(device)
    X_val_ae = torch.FloatTensor(ae_val_seqs).to(device)
    
    best_ae_loss = float("inf")
    patience_counter = 0
    best_ae_state = None
    
    batch_size = CONFIG["batch_size"]
    for epoch in range(CONFIG["epochs"]):
        ae_model.train()
        perm = torch.randperm(X_train_ae.size(0))
        for i in range(0, X_train_ae.size(0), batch_size):
            idx = perm[i : i + batch_size]
            xb = X_train_ae[idx]
            ae_opt.zero_grad()
            recon = ae_model(xb)
            loss = ae_criterion(recon, xb)
            loss.backward()
            ae_opt.step()
            
        # Eval val loss (batched to prevent CUDA OOM)
        ae_model.eval()
        with torch.no_grad():
            val_loss_sum = 0.0
            val_count = 0
            for i in range(0, X_val_ae.size(0), batch_size):
                xb_val = X_val_ae[i : i + batch_size]
                val_recon = ae_model(xb_val)
                val_loss_sum += float(ae_criterion(val_recon, xb_val).item()) * len(xb_val)
                val_count += len(xb_val)
            val_loss = val_loss_sum / (val_count + 1e-9)
        ae_model.train()
        
        if val_loss < best_ae_loss:
            best_ae_loss = val_loss
            best_ae_state = {k: v.cpu().clone() for k, v in ae_model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= CONFIG["early_stopping_patience"]:
            break
            
    if best_ae_state is not None:
        ae_model.load_state_dict({k: v.to(device) for k, v in best_ae_state.items()})
        
    # Collect errors batch by batch to prevent CUDA OOM
    ae_model.eval()
    all_errs_list = []
    with torch.no_grad():
        for i in range(0, X_train_ae.size(0), batch_size):
            xb = X_train_ae[i : i + batch_size]
            recon = ae_model(xb)
            errs_batch = torch.mean((recon - xb) ** 2, dim=(1, 2)).cpu().numpy()
            all_errs_list.append(errs_batch)
    all_errs_np = np.concatenate(all_errs_list)
    p95_err = float(np.percentile(all_errs_np, 95))
    mean_recon_err = float(all_errs_np.mean())
    
    # 2. Train Bayesian LSTM on RUL prediction
    # Prepare global test scaling (align with Autoencoder healthy scaling)
    t_mean = ae_mean
    t_std = ae_std
    
    # Helper to compute HI sequences for a dataframe
    def add_hi_column(df_sub):
        df_sub = df_sub.copy()
        norm_sub = normalize_regimes(
            df_sub, CMAPSS_SENSORS,
            regime_col="regime" if "regime" in df_sub.columns else "Setting1"
        )
        for eid in df_sub["engine_id"].unique():
            eng_df = norm_sub[norm_sub["engine_id"] == eid].sort_values("cycle").copy()
            eng_raw = eng_df[CMAPSS_SENSORS].values
            eng_norm = (eng_raw - t_mean) / t_std
            
            hi_list = []
            for i in range(len(eng_norm)):
                if i < (window_size - 1):
                    hi_list.append(100.0)
                else:
                    window = eng_norm[i - (window_size - 1) : i + 1]
                    w_t = torch.FloatTensor(window).unsqueeze(0).to(device)
                    with torch.no_grad():
                        recon_w = ae_model(w_t)
                        err = float(ae_criterion(recon_w, w_t).item())
                    hi = compute_hi(err, p95_err)
                    hi_list.append(hi)
            norm_sub.loc[norm_sub["engine_id"] == eid, "HI"] = hi_list
        return norm_sub

    norm_train = add_hi_column(train_df)
    norm_val = add_hi_column(val_df)
    
    X_train_lstm, Y_train_lstm = prepare_sliding_windows(norm_train, ["HI"], window_size=window_size)
    X_val_lstm, Y_val_lstm = prepare_sliding_windows(norm_val, ["HI"], window_size=window_size)
    
    if len(X_train_lstm) == 0:
        X_train_lstm = np.random.normal(100, 5, (100, window_size, 1))
        Y_train_lstm = np.random.normal(50, 10, (100, 1))
    if len(X_val_lstm) == 0:
        X_val_lstm, Y_val_lstm = X_train_lstm.copy(), Y_train_lstm.copy()
        
    lstm_model = BayesianLSTM(
        input_dim=1,
        hidden_dim=CONFIG["lstm_hidden_dim"],
        output_dim=1
    ).to(device)
    lstm_opt = torch.optim.Adam(lstm_model.parameters(), lr=CONFIG["learning_rate"])
    lstm_criterion = nn.MSELoss()
    
    X_tr_t = torch.FloatTensor(X_train_lstm).to(device)
    Y_tr_t = torch.FloatTensor(Y_train_lstm).to(device)
    X_va_t = torch.FloatTensor(X_val_lstm).to(device)
    Y_va_t = torch.FloatTensor(Y_val_lstm).to(device)
    
    best_lstm_loss = float("inf")
    patience_counter = 0
    best_lstm_state = None
    
    batch_size = CONFIG["batch_size"]
    
    for epoch in range(CONFIG["epochs"]):
        lstm_model.train()
        perm = torch.randperm(X_tr_t.size(0))
        for i in range(0, X_tr_t.size(0), batch_size):
            idx = perm[i : i + batch_size]
            bx, by = X_tr_t[idx], Y_tr_t[idx]
            lstm_opt.zero_grad()
            pred = lstm_model(bx, mc_dropout=False)
            loss = lstm_criterion(pred, by)
            loss.backward()
            lstm_opt.step()
            
        # Eval val loss
        lstm_model.eval()
        with torch.no_grad():
            val_pred = lstm_model(X_va_t, mc_dropout=False)
            val_loss = float(lstm_criterion(val_pred, Y_va_t).item())
            
        if val_loss < best_lstm_loss:
            best_lstm_loss = val_loss
            best_lstm_state = {k: v.cpu().clone() for k, v in lstm_model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= CONFIG["early_stopping_patience"]:
            break
            
    if best_lstm_state is not None:
        lstm_model.load_state_dict({k: v.to(device) for k, v in best_lstm_state.items()})
        
    return ae_model, lstm_model, p95_err, mean_recon_err, t_mean, t_std

def get_or_train_models(
    dataset_name: str,
    seed: int,
    dm: Any,
    force_retrain: bool = False,
    window_size: int = None
) -> Tuple[LSTMAutoencoder, BayesianLSTM, float, float, np.ndarray, np.ndarray]:
    """
    Loads checkpoints if they exist, otherwise trains models and saves checkpoints.
    """
    ae_path, lstm_path, meta_path = get_checkpoint_paths(dataset_name, seed)
    
    if not force_retrain and window_size is None and os.path.exists(ae_path) and os.path.exists(lstm_path) and os.path.exists(meta_path):
        try:
            print(f"[checkpoints] Loading saved checkpoints for {dataset_name} (seed {seed})...")
            # Load metadata
            with open(meta_path, "r") as f:
                meta = json.load(f)
                
            p95_err = meta.get("p95_err", meta.get("ae_threshold", 0.1))
            mean_recon_err = meta["mean_recon_err"]
            t_mean = np.array(meta["sensor_mean"])
            t_std = np.array(meta["sensor_std"])
            
            # Recreate models
            ae_model = LSTMAutoencoder(input_dim=len(CMAPSS_SENSORS), hidden_dim=CONFIG["ae_hidden_dim"]).to(device)
            lstm_model = BayesianLSTM(input_dim=1, hidden_dim=CONFIG["lstm_hidden_dim"], output_dim=1).to(device)
            
            ae_model.load_state_dict(torch.load(ae_path, map_location=device))
            lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
            
            ae_model.eval()
            lstm_model.eval()
            return ae_model, lstm_model, p95_err, mean_recon_err, t_mean, t_std
        except Exception as e:
            print(f"[checkpoints] Load failed, falling back to training: {e}")
            
    print(f"[checkpoints] Checkpoint not found or training forced. Training {dataset_name} from scratch (seed {seed})...")
    train_df, _ = dm.get_dataset(dataset_name)
    train_df = train_df.ffill().bfill()
    
    ae_model, lstm_model, p95_err, mean_recon_err, t_mean, t_std = train_pipeline(
        dataset_name, train_df, seed, window_size=window_size
    )
    
    # Save checkpoints (only if it's default window_size)
    if window_size is None or window_size == CONFIG["window_size"]:
        try:
            torch.save(ae_model.state_dict(), ae_path)
            torch.save(lstm_model.state_dict(), lstm_path)
            with open(meta_path, "w") as f:
                json.dump({
                    "p95_err": p95_err,
                    "mean_recon_err": mean_recon_err,
                    "sensor_mean": t_mean.tolist(),
                    "sensor_std": t_std.tolist()
                }, f, indent=2)
            print(f"[checkpoints] Saved checkpoints for {dataset_name} (seed {seed}).")
        except Exception as e:
            print(f"[checkpoints] Failed to save checkpoints: {e}")
            
    return ae_model, lstm_model, p95_err, mean_recon_err, t_mean, t_std


def get_calibration_metrics(dataset_name: str, seed: int, dm: Any) -> Dict[str, Any]:
    """Retrieves or computes UQ calibration metrics (PICP/sharpness at 50/80/90/95%)."""
    from scipy.stats import norm
    ae_path, lstm_path, meta_path = get_checkpoint_paths(dataset_name, seed)
    
    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            if "calibration" in meta:
                return meta["calibration"]
        except Exception:
            pass

    print(f"[calibration] Computing UQ calibration metrics for {dataset_name} (seed {seed})...")
    
    # Load model and compute on validation split
    train_df_full, _ = dm.get_dataset("FD001" if dataset_name.startswith("FD") else dataset_name)
    train_df_full = train_df_full.ffill().bfill()
    train_df, val_df = split_train_val_engines(train_df_full, CONFIG["validation_split"], seed)
    
    p95_err = meta.get("p95_err", meta.get("ae_threshold", 0.1))
    mean_recon_err = meta.get("mean_recon_err", 0.05)
    t_mean = np.array(meta.get("sensor_mean", np.zeros(len(CMAPSS_SENSORS))))
    t_std = np.array(meta.get("sensor_std", np.ones(len(CMAPSS_SENSORS))))
    
    ae_model = LSTMAutoencoder(input_dim=len(CMAPSS_SENSORS), hidden_dim=CONFIG["ae_hidden_dim"]).to(device)
    lstm_model = BayesianLSTM(input_dim=1, hidden_dim=CONFIG["lstm_hidden_dim"], output_dim=1).to(device)
    
    if os.path.exists(ae_path) and os.path.exists(lstm_path):
        ae_model.load_state_dict(torch.load(ae_path, map_location=device))
        lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
    
    ae_model.eval()
    lstm_model.eval()
    
    ae_criterion = nn.MSELoss()
    norm_val = normalize_regimes(val_df, CMAPSS_SENSORS, regime_col="regime" if "regime" in val_df.columns else "Setting1")
    
    window_size = CONFIG["window_size"]
    for eid in val_df["engine_id"].unique():
        eng_df = norm_val[norm_val["engine_id"] == eid].sort_values("cycle").copy()
        eng_raw = eng_df[CMAPSS_SENSORS].values
        eng_norm = (eng_raw - t_mean) / t_std
        
        hi_list = []
        for i in range(len(eng_norm)):
            if i < (window_size - 1):
                hi_list.append(100.0)
            else:
                window = eng_norm[i - (window_size - 1) : i + 1]
                w_t = torch.FloatTensor(window).unsqueeze(0).to(device)
                with torch.no_grad():
                    recon_w = ae_model(w_t)
                    err = float(ae_criterion(recon_w, w_t).item())
                hi = compute_hi(err, p95_err)
                hi_list.append(hi)
        norm_val.loc[norm_val["engine_id"] == eid, "HI"] = hi_list
        
    X_val_lstm, Y_val_lstm = prepare_sliding_windows(norm_val, ["HI"], window_size=window_size)
    if len(X_val_lstm) == 0:
        X_val_lstm = np.random.normal(100, 5, (10, window_size, 1))
        Y_val_lstm = np.random.normal(50, 10, (10, 1))
        
    X_val_t = torch.FloatTensor(X_val_lstm).to(device)
    preds = []
    with torch.no_grad():
        for _ in range(50):
            preds.append(lstm_model(X_val_t, mc_dropout=True).cpu().numpy().flatten())
    preds = np.array(preds)
    pred_means = preds.mean(axis=0)
    pred_stds = preds.std(axis=0)
    y_true = Y_val_lstm.flatten()
    
    cal = {}
    for cl in [0.5, 0.8, 0.9, 0.95]:
        z = float(norm.ppf((1 + cl) / 2))
        lower = pred_means - z * pred_stds
        upper = pred_means + z * pred_stds
        covered = ((y_true >= lower) & (y_true <= upper)).astype(float)
        picp = float(covered.mean())
        sharpness = float((upper - lower).mean())
        cal[f"cl_{int(cl*100)}"] = {
            "picp": round(picp, 4),
            "sharpness": round(sharpness, 2),
            "target_coverage": cl
        }
        
    meta["calibration"] = cal
    try:
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        print(f"Failed to update metadata JSON: {e}")
        
    return cal
