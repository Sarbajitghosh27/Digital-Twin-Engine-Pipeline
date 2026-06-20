import os
import sys
import asyncio
import json
import math
from concurrent.futures import ThreadPoolExecutor
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.ensemble import IsolationForest

# Add the root directory to path to import aiml_core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aiml_core.models import LSTMAutoencoder, BayesianLSTM
from aiml_core.explainers import PMAExplainer
from aiml_core.data_loader import DatasetManager, normalize_regimes, prepare_sliding_windows, CMAPSS_SENSORS, NCMAPSS_SENSORS
from aiml_core.benchmark import generate_benchmark_tables, compute_nasa_score, mc_dropout_predict, compute_calibration_metrics

app = FastAPI(title="Prognostics Digital Twin API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- GLOBAL SYSTEM STATES ---
ACTIVE_DATASET = "FD001"
dm = DatasetManager(data_root="data")

# Core AI/ML Models
ae_model: Optional[LSTMAutoencoder] = None
lstm_model: Optional[BayesianLSTM] = None
mean_recon_err: float = 0.5
ae_threshold: float = 0.1
anomaly_model: Optional[IsolationForest] = None

# Telemetry data matrices
test_sensor_mean: Optional[np.ndarray] = None
test_sensor_std: Optional[np.ndarray] = None
train_df_global: Optional[pd.DataFrame] = None
test_df_global: Optional[pd.DataFrame] = None

# Engine State Tracking
ACTIVE_ENGINE_ID = 1

# Load sensor limits config JSON on startup
sensor_limits = {}
try:
    limits_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "sensor_limits.json")
    with open(limits_path, "r", encoding="utf-8") as f:
        sensor_limits = json.load(f)
except Exception as e:
    print(f"Error loading sensor limits in backend: {e}")

class EngineTwinState:
    def __init__(self, engine_id: int, df_subset: pd.DataFrame):
        self.engine_id = engine_id
        self.df_subset = df_subset.sort_values("cycle").copy()
        self.max_cycles = len(self.df_subset)
        self.current_cycle = 1
        self.is_iot_mode = False
        self.iot_data = {}
        self.history: List[Dict] = []
        self.cache: Dict[int, Dict] = {}
        
    def increment(self):
        if not self.is_iot_mode:
            if self.current_cycle < self.max_cycles:
                self.current_cycle += 1
            else:
                self.current_cycle = 1 # Loop back to cycle 1 for continuous simulation
                
    def get_sensors_at_cycle(self, cycle: int) -> Dict[str, float]:
        """Returns raw sensor values at a given cycle."""
        if self.is_iot_mode and "sensors" in self.iot_data:
            return self.iot_data["sensors"]
            
        row = self.df_subset[self.df_subset["cycle"] == cycle]
        if row.empty:
            # Fallback to last cycle if not found
            row = self.df_subset.iloc[-1:]
            
        # Extract sensor list
        sensors = {}
        sensor_list = NCMAPSS_SENSORS if ACTIVE_DATASET == "N-CMAPSS_DS01" else CMAPSS_SENSORS
        for s in sensor_list:
            if s in row.columns:
                sensors[s] = float(row[s].values[0])
            else:
                sensors[s] = 0.0
        return sensors

    def get_components_at_cycle(self, cycle: int) -> Dict[str, float]:
        """Infers individual module health scores based on cycle and degradations."""
        if self.is_iot_mode and "components" in self.iot_data:
            return self.iot_data["components"]
            
        ratio = cycle / self.max_cycles
        # Simulate dynamic module health percentages (100% -> 0%)
        healths = {
            "Fan": max(100 - (35 * (ratio ** 1.8) + random.uniform(-1, 1)), 0.0),
            "LPC": max(100 - (40 * (ratio ** 1.7) + random.uniform(-1, 1)), 0.0),
            "HPC": max(100 - (55 * (ratio ** 2.0) + random.uniform(-1.5, 1.5)), 0.0),
            "Combustor": max(100 - (30 * (ratio ** 1.9) + random.uniform(-1, 1)), 0.0),
            "HPT": max(100 - (60 * (ratio ** 2.2) + random.uniform(-1.5, 1.5)), 0.0),
            "LPT": max(100 - (50 * (ratio ** 2.1) + random.uniform(-1, 1)), 0.0)
        }
        return {k: round(v, 2) for k, v in healths.items()}

# Active fleet database
engines_db: Dict[int, EngineTwinState] = {}

def initialize_models_and_data(dataset_name: str):
    """Loads dataset and trains the Autoencoder, Bayesian LSTM, and Isolation Forest on the fly."""
    global ACTIVE_DATASET, train_df_global, test_df_global, ae_model, lstm_model, mean_recon_err, ae_threshold, anomaly_model, test_sensor_mean, test_sensor_std, engines_db
    
    ACTIVE_DATASET = dataset_name
    print(f"Loading data for {dataset_name}...")
    train_df, test_df = dm.get_dataset(dataset_name)
    
    # Fill missing values if any
    train_df = train_df.ffill().bfill()
    test_df = test_df.ffill().bfill()
    
    train_df_global = train_df
    test_df_global = test_df
    
    sensor_list = CMAPSS_SENSORS
    if dataset_name == "N-CMAPSS_DS01":
        # Overlapping sensors mapped for evaluation
        test_df["FuelFlow"] = test_df["wf"]
        test_df["Setting1"] = test_df["alt"] / 50000.0
        train_df["FuelFlow"] = train_df["wf"]
        train_df["Setting1"] = train_df["alt"] / 50000.0
        
    print("Fitting operating condition normalizer...")
    norm_train = normalize_regimes(train_df, sensor_list, regime_col="regime" if "regime" in train_df.columns else "Setting1")
    norm_test = normalize_regimes(test_df, sensor_list, regime_col="regime" if "regime" in test_df.columns else "Setting1")
    
    # Calculate test_sensor_mean and test_sensor_std using raw train_df before normalization
    X_train_raw = train_df[sensor_list].values
    test_sensor_mean = X_train_raw.mean(axis=0)
    test_sensor_std = X_train_raw.std(axis=0)
    test_sensor_std[test_sensor_std == 0.0] = 1.0
    
    # 1. Fit Anomaly Detection (Isolation Forest) on first 50 cycles of training data (healthy engine states)
    print("Fitting Isolation Forest Anomaly Scorer...")
    healthy_data = norm_train[norm_train["cycle"] <= 50][sensor_list].values
    if len(healthy_data) == 0:
        healthy_data = norm_train[sensor_list].values
    # Fit standard Isolation Forest
    anomaly_model = IsolationForest(n_estimators=30, contamination=0.05, random_state=42)
    anomaly_model.fit(healthy_data)
    
    # 2. Fit Autoencoder for Health Index
    print("Fitting Autoencoder for Health Index...")
    early_df = norm_train[norm_train["cycle"] <= 50].copy()
    early_norm = early_df[sensor_list].values
    
    ae_sequences = []
    for engine_id in early_df["engine_id"].unique():
        eng_data = norm_train[norm_train["engine_id"] == engine_id].sort_values("cycle")[sensor_list].values
        eng_norm = eng_data
        if len(eng_norm) >= 30:
            for i in range(len(eng_norm) - 30 + 1):
                ae_sequences.append(eng_norm[i : i + 30])
                
    ae_sequences = np.array(ae_sequences)
    if len(ae_sequences) == 0:
        ae_sequences = np.random.normal(0, 1, (100, 30, len(sensor_list)))
        
    ae_model = LSTMAutoencoder(input_dim=len(sensor_list), hidden_dim=8).to(device)
    ae_opt = torch.optim.Adam(ae_model.parameters(), lr=0.01)
    ae_criterion = nn.MSELoss()
    
    ae_model.train()
    X_ae_t = torch.FloatTensor(ae_sequences).to(device)
    for epoch in range(3): # Fast 3 epochs training on startup
        ae_opt.zero_grad()
        recon = ae_model(X_ae_t)
        loss = ae_criterion(recon, X_ae_t)
        loss.backward()
        ae_opt.step()
        
    ae_model.eval()
    with torch.no_grad():
        recon_base = ae_model(X_ae_t)
        mean_recon_err = float(ae_criterion(recon_base, X_ae_t).item())
        ae_threshold = max(0.01, mean_recon_err * 2.0)
        
    # 3. Compute HI sequence and fit Bayesian LSTM
    print("Computing Health Index and training Bayesian LSTM...")
    sample_hi_vals = []
    err_offset = mean_recon_err * 0.95
    for engine_id in train_df["engine_id"].unique():
        eng_df = norm_train[norm_train["engine_id"] == engine_id].sort_values("cycle").copy()
        eng_norm = eng_df[sensor_list].values
        
        hi_list = []
        for i in range(len(eng_norm)):
            if i < 29:
                hi_list.append(100.0)
            else:
                window = eng_norm[i - 29 : i + 1]
                window_t = torch.FloatTensor(window).unsqueeze(0).to(device)
                with torch.no_grad():
                    recon_w = ae_model(window_t)
                    err = float(ae_criterion(recon_w, window_t).item())
                hi = 100.0 * np.exp(-max(0.0, err - err_offset) / ae_threshold)
                hi_list.append(hi)
                
        if engine_id == 1:
            sample_hi_vals = hi_list[:10]
            
        norm_train.loc[norm_train["engine_id"] == engine_id, "HI"] = hi_list
        
    print(f"AE Threshold: {ae_threshold:.6f}, Mean Recon Error: {mean_recon_err:.6f}")
    print(f"Sample HI values (first 10 of Engine 1): {[round(x, 2) for x in sample_hi_vals]}")
        
    X_lstm, Y_lstm = prepare_sliding_windows(norm_train, ["HI"], window_size=30)
    
    lstm_model = BayesianLSTM(input_dim=1, hidden_dim=16, output_dim=1).to(device)
    lstm_opt = torch.optim.Adam(lstm_model.parameters(), lr=0.01)
    lstm_criterion = nn.MSELoss()
    
    lstm_model.train()
    X_lstm_t = torch.FloatTensor(X_lstm).to(device)
    Y_lstm_t = torch.FloatTensor(Y_lstm).to(device)
    
    # 3 quick training epochs
    batch_size = 64
    for epoch in range(3):
        permutation = torch.randperm(X_lstm_t.size(0))
        for i in range(0, X_lstm_t.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_lstm_t[indices], Y_lstm_t[indices]
            lstm_opt.zero_grad()
            pred = lstm_model(batch_x, mc_dropout=False)
            loss = lstm_criterion(pred, batch_y)
            loss.backward()
            lstm_opt.step()
            
    # Reset simulation engines database
    engines_db.clear()
    engine_ids = test_df["engine_id"].unique()
    for eid in engine_ids[:20]: # Limit fleet to 20 engines for UI speed
        eid_int = int(eid)
        engines_db[eid_int] = EngineTwinState(eid_int, test_df[test_df["engine_id"] == eid_int])
        
    print(f"Digital Twin Models successfully initialized for {dataset_name}!")

# Run initial load on FD001
initialize_models_and_data("FD001")


# --- CORE INFERENCE LOGIC (MC DROPOUT & PMA SHAP) ---

def run_mc_dropout_prediction(hi_window: np.ndarray, num_samples: int = 30) -> Tuple[float, float, float, float]:
    """
    Performs N Monte Carlo Dropout passes to compute probabilistic RUL estimates.
    Returns:
        P10, Mean (predicted RUL), P90, and standard deviation.
    """
    # hi_window shape: (30, 1)
    # Replicate the window along batch dimension: shape (num_samples, 30, 1)
    x_t = torch.FloatTensor(hi_window).unsqueeze(0).repeat(num_samples, 1, 1).to(device)
    
    lstm_model.eval()
    with torch.no_grad():
        preds = lstm_model(x_t, mc_dropout=True) # shape: (num_samples, 1)
        samples = preds.squeeze().cpu().numpy()
        
    # Compute mean and std
    mean = float(np.mean(samples))
    std = float(np.std(samples))
    
    # Return p10 and p90 as uncertainty bounds, clipped to [0.0, 150.0]
    p10 = float(np.clip(mean - 1.96 * std, 0.0, 150.0))
    p90 = float(np.clip(mean + 1.96 * std, 0.0, 150.0))
    
    return p10, mean, p90, std

def get_engine_metrics_and_explanations(engine_id: int, cycle: int, explain: bool = True) -> Dict:
    """
    Computes real-time ML predictions:
    1. Health Index via LSTM Autoencoder
    2. Probabilistic RUL bounds via MC Dropout LSTM
    3. Anomaly scores via Isolation Forest
    4. SHAP feature attributions via PMA Explainer for both RUL and Anomalies
    """
    engine = engines_db[engine_id]
    
    # Cache key check (only use cache when not in IoT mode)
    if not engine.is_iot_mode and cycle in engine.cache:
        cached_val = engine.cache[cycle]
        if not explain or "explainers" in cached_val:
            return cached_val
            
    sensor_list = CMAPSS_SENSORS
    is_ncmapss = (ACTIVE_DATASET == "N-CMAPSS_DS01")
    
    # Extract the sequence of raw sensor values ending at current cycle (up to window 30)
    cycles_to_load = list(range(max(1, cycle - 29), cycle + 1))
    # Pad if sequence is too short
    while len(cycles_to_load) < 30:
        cycles_to_load.insert(0, cycles_to_load[0])
        
    # Build the sensor sequence matrix
    sensor_sequence = []
    for c in cycles_to_load:
        sens = engine.get_sensors_at_cycle(c)
        # Convert dict to array matching order of CMAPSS_SENSORS
        sensor_sequence.append([sens[k] for k in sensor_list])
        
    sensor_sequence = np.array(sensor_sequence) # Shape: (30, 14)
    
    # Normalize sensors using global scaling parameters
    sensor_seq_norm = (sensor_sequence - test_sensor_mean) / test_sensor_std
    
    # 1. Compute Health Index sequence
    ae_model.eval()
    ae_criterion = nn.MSELoss()
    hi_seq = []
    
    err_offset = mean_recon_err * 0.95
    with torch.no_grad():
        for i in range(len(sensor_seq_norm)):
            # Pad early elements
            if i < 29:
                hi_seq.append(100.0)
            else:
                window_t = torch.FloatTensor(sensor_seq_norm[i-29 : i+1]).unsqueeze(0).to(device)
                recon = ae_model(window_t)
                err = float(ae_criterion(recon, window_t).item())
                hi = 100.0 * np.exp(-max(0.0, err - err_offset) / ae_threshold)
                hi_seq.append(hi)
                
    current_hi = round(hi_seq[-1], 2)
    hi_window = np.array(hi_seq).reshape(-1, 1) # Shape: (30, 1)
    
    # 2. Run MC Dropout for RUL
    p10, p50, p90, std_dev = run_mc_dropout_prediction(hi_window)
    
    # 3. Compute Anomaly Score
    current_sensor_val = sensor_seq_norm[-1].reshape(1, -1)
    raw_anomaly_score = float(anomaly_model.score_samples(current_sensor_val)[0])
    # Map score_samples range (typically -0.8 to -0.3) to 0 - 100% anomaly score
    anomaly_score = max(0.0, min(100.0, (0.45 - raw_anomaly_score) * 180.0))
    
    # Sigmoid failure probability in next 30 cycles
    failure_prob = round(100.0 / (1.0 + math.exp((p50 - 30.0) / 10.0)), 2)
    
    # Map raw sensor values back to output dictionary
    raw_sensors = engine.get_sensors_at_cycle(cycle)
    components = engine.get_components_at_cycle(cycle)
    
    result = {
        "engine_id": engine_id,
        "current_cycle": cycle,
        "max_cycles": engine.max_cycles,
        "sensors": raw_sensors,
        "components": components,
        "predictions": {
            "RUL_predicted": round(p50, 1),
            "RUL_p10": round(p10, 1),
            "RUL_p90": round(p90, 1),
            "rul_mean": round(p50, 1),
            "rul_lower": round(p10, 1),
            "rul_upper": round(p90, 1),
            "hi_uncertainty": round(std_dev, 2),
            "HealthIndex": round(current_hi, 1),
            "AnomalyScore": round(anomaly_score, 1),
            "FailureProbability": round(failure_prob, 2),
            "is_anomalous": anomaly_score > 75.0
        }
    }
    
    # 4. PMA EXPLANATIONS (SHAP)
    if explain:
        # Baseline normal state (mean of healthy sensors)
        baseline_sensor_val = np.zeros((1, len(sensor_list))) # Normalized baseline is zeros
        
        # Anomaly SHAP Attribution
        def anomaly_scorer_func(x):
            # x shape (batch, num_sensors)
            scores = anomaly_model.score_samples(x)
            return np.maximum(0.0, np.minimum(100.0, (0.45 - scores) * 180.0))
            
        anomaly_explainer = PMAExplainer(anomaly_scorer_func, baseline_sensor_val)
        anomaly_shaps = anomaly_explainer.explain(current_sensor_val)
        
        # RUL SHAP Attribution (explain how current sensors reduce the RUL from baseline)
        # We define a function mapping raw sensor window to predicted RUL
        def sensor_to_rul_func(x_norm_3d):
            # x_norm_3d shape: (1, 30, 14)
            x_norm_t = torch.FloatTensor(x_norm_3d).to(device)
            with torch.no_grad():
                recon_3d = ae_model(x_norm_t)
                errs = torch.mean((recon_3d - x_norm_t) ** 2, dim=2).squeeze(0).cpu().numpy()
                
            hi_list = []
            err_offset_shap = mean_recon_err * 0.95
            for e in errs:
                hi_list.append(100.0 * np.exp(-max(0.0, e - err_offset_shap) / ae_threshold))
                
            hi_arr_t = torch.FloatTensor(hi_list).unsqueeze(0).unsqueeze(2).to(device)
            with torch.no_grad():
                pred_r = lstm_model(hi_arr_t, mc_dropout=False)
            return float(pred_r.item())
            
        baseline_sensor_seq = np.zeros((1, 30, len(sensor_list))) # fully normalized healthy baseline
        current_sensor_seq = sensor_seq_norm.reshape(1, 30, -1)
        
        rul_explainer = PMAExplainer(sensor_to_rul_func, baseline_sensor_seq)
        rul_shaps = rul_explainer.explain(current_sensor_seq)
        
        # Rank top 3 sensors for anomaly
        sensor_shap_pairs = list(zip(sensor_list, anomaly_shaps))
        sensor_shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        top_drivers = [{"sensor": k, "val": round(v, 2)} for k, v in sensor_shap_pairs[:3]]
        
        result["explainers"] = {
            "anomaly_shap": {k: float(v) for k, v in zip(sensor_list, anomaly_shaps)},
            "rul_shap": {k: float(v) for k, v in zip(sensor_list, rul_shaps)},
            "top_anomaly_drivers": top_drivers
        }
    else:
        result["explainers"] = {}
        
    if not engine.is_iot_mode:
        engine.cache[cycle] = result
        
    return result


# --- SIMULATION CONTROL ---
class SimManager:
    def __init__(self):
        self.is_running = True
        self.speed = 1.0  # seconds per cycle
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        dead = []
        for conn in self.active_connections:
            try:
                await conn.send_text(json.dumps(message))
            except Exception:
                dead.append(conn)
        for conn in dead:
            self.disconnect(conn)

    async def run_loop(self):
        loop = asyncio.get_event_loop()
        while True:
            if self.is_running:
                def _compute_tick():
                    updates = []
                    for engine_id, eng in engines_db.items():
                        if not eng.is_iot_mode:
                            eng.increment()
                            try:
                                status = get_engine_metrics_and_explanations(
                                    engine_id, eng.current_cycle, explain=(engine_id == ACTIVE_ENGINE_ID)
                                )
                                updates.append(status)
                            except Exception as e:
                                print(f"Error computing engine status in sim: {e}")
                    return updates
                
                updates = await loop.run_in_executor(_thread_pool, _compute_tick)
                
                if updates:
                    await self.broadcast({
                        "type": "telemetry_update",
                        "timestamp": time.time(),
                        "engines": {u["engine_id"]: u for u in updates}
                    })
            await asyncio.sleep(self.speed)

sim_manager = SimManager()

# Thread pool for running CPU-bound inference without blocking the event loop
_thread_pool = ThreadPoolExecutor(max_workers=4)

# --- REST API ENDPOINTS ---

class IoTTelemetryInput(BaseModel):
    engine_id: int
    cycle: int
    sensors: Dict[str, float]
    components: Dict[str, float]
    predictions: Dict[str, float]

@app.get("/")
async def get_root():
    return {
        "status": "online",
        "service": "Aero-Twin Predictive Inference Engine API",
        "active_dataset": ACTIVE_DATASET,
        "engines_monitored": len(engines_db)
    }

@app.post("/api/dataset/select")
def select_dataset(payload: dict):
    dataset_name = payload.get("dataset", "FD001")
    if dataset_name not in ["FD001", "FD002", "FD003", "FD004", "N-CMAPSS_DS01"]:
        raise HTTPException(status_code=400, detail="Invalid dataset selected")
        
    initialize_models_and_data(dataset_name)
    return {"status": "success", "active_dataset": ACTIVE_DATASET}

@app.get("/api/fleet/summary")
async def get_fleet_summary():
    loop = asyncio.get_event_loop()
    def _compute():
        total_engines = len(engines_db)
        if total_engines == 0:
            return {
                "total_engines": 0,
                "fleet_health": 0.0,
                "average_rul": 0.0,
                "active_alerts": 0,
                "simulation_speed": sim_manager.speed,
                "is_running": sim_manager.is_running,
                "active_dataset": ACTIVE_DATASET
            }

        total_rul = 0.0
        total_health = 0.0
        active_alerts = 0
        successful_engines = 0

        for engine_id, eng in engines_db.items():
            try:
                status = get_engine_metrics_and_explanations(engine_id, eng.current_cycle, explain=False)
                pred = status["predictions"]
                total_rul += pred["RUL_predicted"]
                total_health += pred["HealthIndex"]
                successful_engines += 1

                # Check critical sensor limit violations
                has_critical = False
                dataset_limits = sensor_limits.get(ACTIVE_DATASET)
                if dataset_limits:
                    for s_key, val in status["sensors"].items():
                        metadata = dataset_limits.get(s_key)
                        if metadata:
                            if metadata.get("reverse"):
                                if val <= metadata["threshold"] * 0.96:
                                    has_critical = True
                                    break
                            else:
                                if val >= metadata["threshold"] * 1.04:
                                    has_critical = True
                                    break

                if pred["is_anomalous"] or pred["RUL_predicted"] < 35.0 or has_critical:
                    active_alerts += 1
            except Exception:
                pass

        return {
            "total_engines": total_engines,
            "fleet_health": round(total_health / successful_engines, 1) if successful_engines > 0 else 0.0,
            "average_rul": round(total_rul / successful_engines, 1) if successful_engines > 0 else 0.0,
            "active_alerts": active_alerts,
            "simulation_speed": sim_manager.speed,
            "is_running": sim_manager.is_running,
            "active_dataset": ACTIVE_DATASET
        }
    return await loop.run_in_executor(_thread_pool, _compute)

@app.get("/api/engines")
def get_engines_list():
    return [
        {"engine_id": eid, "current_cycle": eng.current_cycle, "max_cycles": eng.max_cycles, "is_iot_mode": eng.is_iot_mode}
        for eid, eng in engines_db.items()
    ]

@app.get("/api/engines/{engine_id}/status")
def get_engine_status(engine_id: int):
    if engine_id not in engines_db:
        raise HTTPException(status_code=404, detail="Engine twin not found")
    eng = engines_db[engine_id]
    global ACTIVE_ENGINE_ID
    ACTIVE_ENGINE_ID = engine_id
    return get_engine_metrics_and_explanations(engine_id, eng.current_cycle, explain=True)

@app.get("/api/engines/{engine_id}/cycle/{cycle}")
@app.post("/api/engines/{engine_id}/cycle/{cycle}")
def set_engine_cycle(engine_id: int, cycle: int):
    if engine_id not in engines_db:
        raise HTTPException(status_code=404, detail="Engine twin not found")
    eng = engines_db[engine_id]
    if cycle < 1 or cycle > eng.max_cycles:
        raise HTTPException(status_code=400, detail="Invalid cycle number")
    eng.current_cycle = cycle
    global ACTIVE_ENGINE_ID
    ACTIVE_ENGINE_ID = engine_id
    return get_engine_metrics_and_explanations(engine_id, cycle, explain=True)

@app.get("/api/engines/{engine_id}/history")
def get_engine_history(engine_id: int, cycle: Optional[int] = None):
    if engine_id not in engines_db:
        raise HTTPException(status_code=404, detail="Engine twin not found")
    eng = engines_db[engine_id]
    end_cycle = cycle if cycle is not None else eng.current_cycle
    history = []
    for c in range(1, end_cycle + 1):
        try:
            history.append(get_engine_metrics_and_explanations(engine_id, c, explain=False))
        except Exception:
            pass
    return history

@app.get("/api/engines/{engine_id}/prediction")
def get_engine_future_projection(engine_id: int):
    """Generates the future remaining lifetime trajectory forecast with uncertainty bands."""
    if engine_id not in engines_db:
        raise HTTPException(status_code=404, detail="Engine twin not found")
    eng = engines_db[engine_id]
    
    future = []
    # Project 40 cycles forward or up to end-of-life
    start_cycle = eng.current_cycle
    end_cycle = min(eng.max_cycles, start_cycle + 45)
    
    for c in range(start_cycle, end_cycle + 1):
        try:
            status = get_engine_metrics_and_explanations(engine_id, c)
            future.append({
                "cycle": c,
                "predictions": status["predictions"],
                "explainers": status["explainers"]
            })
        except Exception:
            pass
    return future

@app.get("/api/alerts")
def get_alerts():
    alerts_list = []
    for eid, eng in engines_db.items():
        try:
            status = get_engine_metrics_and_explanations(eid, eng.current_cycle, explain=False)
            pred = status["predictions"]
            
            # Anomaly alert
            if pred["AnomalyScore"] > 70.0:
                status_full = get_engine_metrics_and_explanations(eid, eng.current_cycle, explain=True)
                top_3 = [d["sensor"] for d in status_full["explainers"]["top_anomaly_drivers"]]
                alerts_list.append({
                    "engine_id": eid,
                    "severity": "critical" if pred["AnomalyScore"] > 85.0 else "warning",
                    "type": "Anomaly Alert",
                    "message": f"High Anomaly Score: {pred['AnomalyScore']}%. Drivers: {', '.join(top_3)}",
                    "timestamp": time.time()
                })
                
            # Low RUL alert
            if pred["RUL_predicted"] < 35.0:
                alerts_list.append({
                    "engine_id": eid,
                    "severity": "critical" if pred["RUL_predicted"] < 15.0 else "warning",
                    "type": "Maintenance Due",
                    "message": f"Critical Remaining Useful Life: {pred['RUL_predicted']} cycles [{pred['RUL_p10']} - {pred['RUL_p90']}]",
                    "timestamp": time.time()
                })
        except Exception:
            pass
            
    alerts_list.sort(key=lambda x: (0 if x["severity"] == "critical" else 1, -x["timestamp"]))
    return alerts_list[:30]

@app.post("/api/telemetry")
async def post_iot_telemetry(data: IoTTelemetryInput):
    eid = data.engine_id
    if eid not in engines_db:
        # Create dynamically from fallback df
        train_df, test_df = dm.get_dataset(ACTIVE_DATASET)
        engines_db[eid] = EngineTwinState(eid, test_df)
        
    eng = engines_db[eid]
    eng.is_iot_mode = True
    eng.current_cycle = data.cycle
    eng.iot_data = {
        "sensors": data.sensors,
        "components": data.components,
        "predictions": data.predictions
    }
    
    status = get_engine_metrics_and_explanations(eid, data.cycle)
    await sim_manager.broadcast({
        "type": "telemetry_update",
        "timestamp": time.time(),
        "engines": {eid: status}
    })
    
    return {"status": "success", "engine_id": eid, "mode": "IoT Live Stream Ingestion"}

@app.post("/api/simulation/control")
def post_sim_control(control: dict):
    if "is_running" in control:
        sim_manager.is_running = bool(control["is_running"])
    if "speed" in control:
        sim_manager.speed = max(0.1, float(control["speed"]))
    if "reset" in control and control["reset"]:
        for eid in engines_db:
            engines_db[eid].current_cycle = 1
            engines_db[eid].is_iot_mode = False
    if "clear_iot" in control and control["clear_iot"]:
        for eid in engines_db.values():
            eid.is_iot_mode = False
            eid.iot_data = {}
            
    return {
        "is_running": sim_manager.is_running,
        "speed": sim_manager.speed,
        "engines_active": len(engines_db)
    }

# Benchmark execution API
@app.get("/api/research/benchmark")
def get_benchmark_results():
    """Triggers the cross-dataset transfer generalization benchmark suite.
    Returns all research metrics: RMSE, NASA Score, PICP, Sharpness, PMA AUDC,
    baseline comparisons, ablation results, and calibration reliability data.
    """
    try:
        results = generate_benchmark_tables()
        return {
            "status": "success",
            "latex": results["latex"],
            "markdown": results["markdown"],
            "results": results["results"],
            "reliability_data": results.get("reliability_data", {}),
            "pma_attributions": results.get("pma_attributions", {}),
            "faithfulness": results.get("faithfulness", {}),
            "baselines": results.get("baselines", {}),
            "ablation": results.get("ablation", {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark execution failed: {e}")

@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    await sim_manager.connect(websocket)
    try:
        # Initial status broadcast
        initial_states = {}
        for eid, eng in engines_db.items():
            try:
                initial_states[eid] = get_engine_metrics_and_explanations(eid, eng.current_cycle)
            except Exception:
                pass
                
        await websocket.send_text(json.dumps({
            "type": "initial_state",
            "timestamp": time.time(),
            "engines": initial_states
        }))
        
        while True:
            # Maintain connection
            await websocket.receive_text()
    except WebSocketDisconnect:
        sim_manager.disconnect(websocket)
    except Exception:
        sim_manager.disconnect(websocket)

# Startup tasks — use lifespan instead of the deprecated on_event
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app_instance):
    # Start the simulation broadcast loop on startup
    asyncio.create_task(sim_manager.run_loop())
    yield
    # Cleanup on shutdown (nothing to do currently)

app.router.lifespan_context = lifespan

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
