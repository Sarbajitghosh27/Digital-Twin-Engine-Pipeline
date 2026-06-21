import os
import sys
import asyncio
import json
import math
import hashlib
from concurrent.futures import ThreadPoolExecutor
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.ensemble import IsolationForest

# Add the root directory to path to import aiml_core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aiml_core.models import LSTMAutoencoder, BayesianLSTM
from aiml_core.explainers import PMAExplainer
from aiml_core.data_loader import DatasetManager, normalize_regimes, prepare_sliding_windows, CMAPSS_SENSORS, NCMAPSS_SENSORS
from aiml_core.config import CONFIG
from aiml_core.train_utils import get_or_train_models, get_checkpoint_paths, get_calibration_metrics
from aiml_core.benchmark import generate_benchmark_tables, compute_nasa_score, mc_dropout_predict, compute_calibration_metrics
from aiml_core.faithfulness import compute_integrated_gradients_attributions

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

# Explainability cache & JSON persistence
EXPLAINABILITY_CACHE: Dict[str, Dict] = {}

# Fleet summary memory cache
FLEET_SUMMARY_CACHE: Optional[Dict] = None
FLEET_SUMMARY_CACHE_SIGNATURE: str = ""

# Load sensor limits config JSON on startup
sensor_limits = {}
try:
    limits_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "sensor_limits.json")
    with open(limits_path, "r", encoding="utf-8") as f:
        sensor_limits = json.load(f)
except Exception as e:
    print(f"Error loading sensor limits in backend: {e}")


def get_file_sha256(filepath: str) -> str:
    """Computes the SHA256 hash of a file's actual weight contents to avoid redundant invalidations."""
    if not os.path.exists(filepath):
        return "not_found"
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_active_models_hash(dataset_name: str) -> str:
    """Retrieves combined SHA256 hash of the active AE and LSTM model weights checkpoints."""
    ae_path, lstm_path, _ = get_checkpoint_paths(dataset_name, 42)
    ae_hash = get_file_sha256(ae_path)
    lstm_hash = get_file_sha256(lstm_path)
    return f"{ae_hash}_{lstm_hash}"


def load_explainability_cache():
    """Loads the offline Integrated Gradients cache from JSON."""
    global EXPLAINABILITY_CACHE
    cache_path = os.path.join(CONFIG["checkpoints_dir"], "explainability_cache.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                EXPLAINABILITY_CACHE = json.load(f)
            print(f"[cache] Loaded {len(EXPLAINABILITY_CACHE)} cached explainability entries.")
        except Exception as e:
            print(f"[cache] Failed to load explainability cache: {e}")
            EXPLAINABILITY_CACHE = {}


def save_explainability_cache():
    """Saves the explainability cache to JSON."""
    cache_path = os.path.join(CONFIG["checkpoints_dir"], "explainability_cache.json")
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(EXPLAINABILITY_CACHE, f, indent=2)
    except Exception as e:
        print(f"[cache] Failed to save explainability cache: {e}")


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
    """Loads dataset and loads pre-trained checkpoints (or trains them if missing)."""
    global ACTIVE_DATASET, train_df_global, test_df_global, ae_model, lstm_model, mean_recon_err, ae_threshold, anomaly_model, test_sensor_mean, test_sensor_std, engines_db, FLEET_SUMMARY_CACHE_SIGNATURE
    
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
        test_df["FuelFlow"] = test_df["wf"]
        test_df["Setting1"] = test_df["alt"] / 50000.0
        train_df["FuelFlow"] = train_df["wf"]
        train_df["Setting1"] = train_df["alt"] / 50000.0
        
    print("Fitting operating condition normalizer...")
    norm_train = normalize_regimes(train_df, sensor_list, regime_col="regime" if "regime" in train_df.columns else "Setting1")
    
    # 1. Load/train AE and LSTM models using get_or_train_models with seed=42 (default for dashboard)
    ae_model, lstm_model, ae_threshold, mean_recon_err, test_sensor_mean, test_sensor_std = get_or_train_models(
        dataset_name, seed=42, dm=dm, force_retrain=False
    )
    
    # Load or compute explainability JSON cache
    load_explainability_cache()
    
    # 2. Fit Anomaly Detection (Isolation Forest) on first 50 cycles of training data (healthy engine states)
    print("Fitting Isolation Forest Anomaly Scorer...")
    healthy_data = norm_train[norm_train["cycle"] <= 50][sensor_list].values
    if len(healthy_data) == 0:
        healthy_data = norm_train[sensor_list].values
    anomaly_model = IsolationForest(n_estimators=30, contamination=0.05, random_state=42)
    anomaly_model.fit(healthy_data)
    
    # 3. Reset simulation engines database
    engines_db.clear()
    engine_ids = test_df["engine_id"].unique()
    for eid in engine_ids[:20]: # Limit fleet to 20 engines for UI speed
        eid_int = int(eid)
        engines_db[eid_int] = EngineTwinState(eid_int, test_df[test_df["engine_id"] == eid_int])
        
    FLEET_SUMMARY_CACHE_SIGNATURE = "" # Reset memory cache
    print(f"Digital Twin Models successfully initialized for {dataset_name}!")

# Run initial load on FD001
initialize_models_and_data("FD001")


# --- CORE INFERENCE LOGIC (MC DROPOUT & INTEGRATED GRADIENTS CACHING) ---

def run_mc_dropout_prediction(hi_window: np.ndarray, num_samples: int = 30) -> Tuple[float, float, float, float]:
    """
    Performs N Monte Carlo Dropout passes to compute probabilistic RUL estimates.
    Returns:
        P10, Mean (predicted RUL), P90, and standard deviation.
    """
    x_t = torch.FloatTensor(hi_window).unsqueeze(0).repeat(num_samples, 1, 1).to(device)
    
    lstm_model.eval()
    with torch.no_grad():
        preds = lstm_model(x_t, mc_dropout=True) # shape: (num_samples, 1)
        samples = preds.squeeze().cpu().numpy()
        
    mean = float(np.mean(samples))
    std = float(np.std(samples))
    
    p10 = float(np.clip(mean - 1.96 * std, 0.0, 150.0))
    p90 = float(np.clip(mean + 1.96 * std, 0.0, 150.0))
    
    return p10, mean, p90, std

def get_engine_metrics_and_explanations(engine_id: int, cycle: int, explain: bool = True) -> Dict:
    """
    Computes real-time ML predictions:
    1. Health Index via LSTM Autoencoder
    2. Probabilistic RUL bounds via MC Dropout LSTM
    3. Anomaly scores via Isolation Forest
    4. Integrated Gradients attributions cached dynamically.
    """
    engine = engines_db[engine_id]
    
    # Cache key check (only use local state cache when not in IoT mode)
    if not engine.is_iot_mode and cycle in engine.cache:
        cached_val = engine.cache[cycle]
        if not explain or "explainers" in cached_val:
            return cached_val
            
    sensor_list = CMAPSS_SENSORS
    
    # Extract the sequence of raw sensor values ending at current cycle (up to window 30)
    cycles_to_load = list(range(max(1, cycle - 29), cycle + 1))
    while len(cycles_to_load) < 30:
        cycles_to_load.insert(0, cycles_to_load[0])
        
    sensor_sequence = []
    for c in cycles_to_load:
        sens = engine.get_sensors_at_cycle(c)
        sensor_sequence.append([sens[k] for k in sensor_list])
        
    sensor_sequence = np.array(sensor_sequence) # Shape: (30, 14)
    
    sensor_seq_norm = (sensor_sequence - test_sensor_mean) / test_sensor_std
    
    # 1. Compute Health Index sequence
    ae_model.eval()
    ae_criterion = nn.MSELoss()
    hi_seq = []
    
    err_offset = mean_recon_err * 0.95
    with torch.no_grad():
        for i in range(len(sensor_seq_norm)):
            if i < 29:
                hi_seq.append(100.0)
            else:
                window_t = torch.FloatTensor(sensor_seq_norm[i-29 : i+1]).unsqueeze(0).to(device)
                recon = ae_model(window_t)
                err = float(ae_criterion(recon, window_t).item())
                hi = 100.0 * np.exp(-max(0.0, err - err_offset) / ae_threshold)
                hi_seq.append(hi)
                
    current_hi = round(hi_seq[-1], 2)
    hi_window = np.array(hi_seq).reshape(-1, 1)
    
    # 2. Run MC Dropout for RUL
    p10, p50, p90, std_dev = run_mc_dropout_prediction(hi_window)
    
    # 3. Compute Anomaly Score
    current_sensor_val = sensor_seq_norm[-1].reshape(1, -1)
    raw_anomaly_score = float(anomaly_model.score_samples(current_sensor_val)[0])
    anomaly_score = max(0.0, min(100.0, (0.45 - raw_anomaly_score) * 180.0))
    
    failure_prob = round(100.0 / (1.0 + math.exp((p50 - 30.0) / 10.0)), 2)
    
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
    
    # 4. CACHED INTEGRATED GRADIENTS & SHAP EXPLANATIONS
    if explain:
        # Check explainability JSON cache
        model_hash = get_active_models_hash(ACTIVE_DATASET)
        cache_key = f"{model_hash}_{engine_id}_{cycle}"
        
        if cache_key in EXPLAINABILITY_CACHE:
            result["explainers"] = EXPLAINABILITY_CACHE[cache_key]
        else:
            # Baseline normal state (mean of healthy sensors)
            baseline_sensor_val = np.zeros((1, len(sensor_list)))
            
            # Anomaly SHAP Attribution via PMA (Isolation Forest)
            def anomaly_scorer_func(x):
                scores = anomaly_model.score_samples(x)
                return np.maximum(0.0, np.minimum(100.0, (0.45 - scores) * 180.0))
                
            anomaly_explainer = PMAExplainer(anomaly_scorer_func, baseline_sensor_val)
            anomaly_shaps = anomaly_explainer.explain(current_sensor_val)
            
            # True Integrated Gradients for Deep-Learning RUL predictor
            current_sensor_seq = sensor_seq_norm.reshape(1, 30, -1)
            try:
                rul_shaps = compute_integrated_gradients_attributions(
                    ae_model, lstm_model, current_sensor_seq, ae_threshold, mean_recon_err, steps=25
                )
            except Exception as e:
                print(f"[IG] Failed to compute Integrated Gradients, falling back to zeros: {e}")
                rul_shaps = np.zeros(len(sensor_list))
            
            # Top Anomaly Drivers
            sensor_shap_pairs = list(zip(sensor_list, anomaly_shaps))
            sensor_shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
            top_drivers = [{"sensor": k, "val": round(v, 2)} for k, v in sensor_shap_pairs[:3]]
            
            explainers_dict = {
                "anomaly_shap": {k: float(v) for k, v in zip(sensor_list, anomaly_shaps)},
                "rul_shap": {k: float(v) for k, v in zip(sensor_list, rul_shaps)},
                "top_anomaly_drivers": top_drivers
            }
            
            # Persist to cache
            EXPLAINABILITY_CACHE[cache_key] = explainers_dict
            save_explainability_cache()
            
            result["explainers"] = explainers_dict
    else:
        result["explainers"] = {
            "anomaly_shap": {},
            "rul_shap": {},
            "top_anomaly_drivers": []
        }
        
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
        if websocket in self.active_connections:
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
_thread_pool = ThreadPoolExecutor(max_workers=4)


# --- REST API ENDPOINTS ---

class IoTTelemetryInput(BaseModel):
    engine_id: int
    cycle: int
    sensors: Dict[str, float]
    components: Dict[str, float]
    predictions: Dict[str, float]


# --- UNVERSIONED LEGACY ENDPOINTS (307 REDIRECTS FOR GET, 410 GONE FOR POST) ---

@app.get("/api/fleet/summary")
async def legacy_fleet_summary():
    return RedirectResponse(url="/api/v1/fleet/summary", status_code=307)

@app.get("/api/engines")
async def legacy_engines_list():
    return RedirectResponse(url="/api/v1/engines", status_code=307)

@app.get("/api/engines/{engine_id}/status")
async def legacy_engine_status(engine_id: int):
    return RedirectResponse(url=f"/api/v1/predict/{engine_id}/cycle/last", status_code=307)

@app.get("/api/engines/{engine_id}/cycle/{cycle}")
async def legacy_get_engine_cycle(engine_id: int, cycle: int):
    return RedirectResponse(url=f"/api/v1/predict/{engine_id}/cycle/{cycle}", status_code=307)

@app.post("/api/engines/{engine_id}/cycle/{cycle}")
async def legacy_post_engine_cycle(engine_id: int, cycle: int):
    return JSONResponse(
        status_code=410,
        content={"detail": "Unversioned POST endpoints are deprecated. Please post to /api/v1/engines/{engine_id}/cycle/{cycle} instead."}
    )

@app.get("/api/engines/{engine_id}/history")
async def legacy_engine_history(engine_id: int, cycle: Optional[int] = None):
    url = f"/api/v1/engines/{engine_id}/history"
    if cycle is not None:
        url += f"?cycle={cycle}"
    return RedirectResponse(url=url, status_code=307)

@app.get("/api/engines/{engine_id}/prediction")
async def legacy_engine_prediction(engine_id: int):
    return RedirectResponse(url=f"/api/v1/engines/{engine_id}/prediction", status_code=307)

@app.get("/api/alerts")
async def legacy_alerts():
    return RedirectResponse(url="/api/v1/alerts", status_code=307)

@app.post("/api/telemetry")
async def legacy_telemetry(data: dict):
    return JSONResponse(
        status_code=410,
        content={"detail": "Unversioned POST endpoints are deprecated. Please post to /api/v1/telemetry instead."}
    )

@app.post("/api/simulation/control")
async def legacy_sim_control(control: dict):
    return JSONResponse(
        status_code=410,
        content={"detail": "Unversioned POST endpoints are deprecated. Please post to /api/v1/simulation/control instead."}
    )

@app.post("/api/dataset/select")
async def legacy_dataset_select(payload: dict):
    return JSONResponse(
        status_code=410,
        content={"detail": "Unversioned POST endpoints are deprecated. Please post to /api/v1/dataset/select instead."}
    )

@app.get("/api/research/benchmark")
async def legacy_benchmark():
    return RedirectResponse(url="/api/v1/research/benchmark", status_code=307)


# --- VERSIONED API V1 ENDPOINTS ---

@app.get("/")
@app.get("/api/v1")
async def get_root():
    return {
        "status": "online",
        "service": "Aero-Twin Predictive Inference Engine API",
        "version": "v1",
        "active_dataset": ACTIVE_DATASET,
        "engines_monitored": len(engines_db)
    }

@app.post("/api/v1/dataset/select")
def select_dataset_v1(payload: dict):
    dataset_name = payload.get("dataset", "FD001")
    if dataset_name not in ["FD001", "FD002", "FD003", "FD004", "N-CMAPSS_DS01"]:
        raise HTTPException(status_code=400, detail="Invalid dataset selected")
    initialize_models_and_data(dataset_name)
    return {"status": "success", "active_dataset": ACTIVE_DATASET}

@app.get("/api/v1/fleet/summary")
async def get_fleet_summary_v1():
    """
    Computes fleet health summary with thread safety and caching.
    Uses current engine cycles state as signature to invalidate the cache.
    """
    global FLEET_SUMMARY_CACHE, FLEET_SUMMARY_CACHE_SIGNATURE
    
    # Calculate state signature of all engine cycles
    current_signature = "_".join(
        f"{eid}:{eng.current_cycle}:{eng.is_iot_mode}" 
        for eid, eng in sorted(engines_db.items())
    )
    
    if FLEET_SUMMARY_CACHE is not None and current_signature == FLEET_SUMMARY_CACHE_SIGNATURE:
        return FLEET_SUMMARY_CACHE

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
        
    res = await loop.run_in_executor(_thread_pool, _compute)
    FLEET_SUMMARY_CACHE = res
    FLEET_SUMMARY_CACHE_SIGNATURE = current_signature
    return res

@app.get("/api/v1/engines")
def get_engines_list_v1():
    return [
        {"engine_id": eid, "current_cycle": eng.current_cycle, "max_cycles": eng.max_cycles, "is_iot_mode": eng.is_iot_mode}
        for eid, eng in sorted(engines_db.items())
    ]

@app.get("/api/v1/predict/{engine_id}/cycle/{cycle}")
@app.post("/api/v1/predict/{engine_id}/cycle/{cycle}")
def get_predict_v1(engine_id: int, cycle: str):
    """Returns predictions (Health Index, RUL, Anomaly) for engine at cycle."""
    if engine_id not in engines_db:
        raise HTTPException(status_code=404, detail="Engine twin not found")
    eng = engines_db[engine_id]
    
    if cycle == "last":
        target_cycle = eng.current_cycle
    else:
        try:
            target_cycle = int(cycle)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid cycle path parameter")
            
    if target_cycle < 1 or target_cycle > eng.max_cycles:
        raise HTTPException(status_code=400, detail="Invalid cycle number")
        
    # Update active engine tracker
    global ACTIVE_ENGINE_ID
    ACTIVE_ENGINE_ID = engine_id
    eng.current_cycle = target_cycle
    
    return get_engine_metrics_and_explanations(engine_id, target_cycle, explain=True)

@app.get("/api/v1/explain/{engine_id}/cycle/{cycle}")
def get_explain_v1(engine_id: int, cycle: str):
    """Returns attributions (SHAP values & Integrated Gradients) for engine at cycle."""
    if engine_id not in engines_db:
        raise HTTPException(status_code=404, detail="Engine twin not found")
    eng = engines_db[engine_id]
    
    if cycle == "last":
        target_cycle = eng.current_cycle
    else:
        try:
            target_cycle = int(cycle)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid cycle path parameter")
            
    status = get_engine_metrics_and_explanations(engine_id, target_cycle, explain=True)
    return status["explainers"]

@app.get("/api/v1/uncertainty/{engine_id}/cycle/{cycle}")
def get_uncertainty_v1(engine_id: int, cycle: str):
    """Returns MC-Dropout sample bounds and UQ calibration metrics (PICP/sharpness)."""
    if engine_id not in engines_db:
        raise HTTPException(status_code=404, detail="Engine twin not found")
    eng = engines_db[engine_id]
    
    if cycle == "last":
        target_cycle = eng.current_cycle
    else:
        try:
            target_cycle = int(cycle)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid cycle path parameter")
            
    status = get_engine_metrics_and_explanations(engine_id, target_cycle, explain=False)
    preds = status["predictions"]
    
    # Load or compute dataset calibration metrics
    cal_metrics = get_calibration_metrics(ACTIVE_DATASET, seed=42, dm=dm)
    
    return {
        "engine_id": engine_id,
        "cycle": target_cycle,
        "rul_predicted": preds["RUL_predicted"],
        "rul_p10": preds["RUL_p10"],
        "rul_p90": preds["RUL_p90"],
        "hi_uncertainty_std": preds["hi_uncertainty"],
        "calibration": cal_metrics
    }

@app.get("/api/v1/engines/{engine_id}/history")
def get_engine_history_v1(engine_id: int, cycle: Optional[int] = None):
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

@app.get("/api/v1/engines/{engine_id}/prediction")
def get_engine_future_projection_v1(engine_id: int):
    if engine_id not in engines_db:
        raise HTTPException(status_code=404, detail="Engine twin not found")
    eng = engines_db[engine_id]
    
    future = []
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

@app.get("/api/v1/alerts")
def get_alerts_v1():
    alerts_list = []
    for eid, eng in engines_db.items():
        try:
            status = get_engine_metrics_and_explanations(eid, eng.current_cycle, explain=False)
            pred = status["predictions"]
            
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

@app.post("/api/v1/telemetry")
async def post_iot_telemetry_v1(data: IoTTelemetryInput):
    global FLEET_SUMMARY_CACHE_SIGNATURE
    eid = data.engine_id
    if eid not in engines_db:
        _, test_df = dm.get_dataset(ACTIVE_DATASET)
        engines_db[eid] = EngineTwinState(eid, test_df)
        
    eng = engines_db[eid]
    eng.is_iot_mode = True
    eng.current_cycle = data.cycle
    eng.iot_data = {
        "sensors": data.sensors,
        "components": data.components,
        "predictions": data.predictions
    }
    
    # Invalidate cache
    FLEET_SUMMARY_CACHE_SIGNATURE = ""
    
    status = get_engine_metrics_and_explanations(eid, data.cycle)
    await sim_manager.broadcast({
        "type": "telemetry_update",
        "timestamp": time.time(),
        "engines": {eid: status}
    })
    
    return {"status": "success", "engine_id": eid, "mode": "IoT Live Stream Ingestion"}

@app.post("/api/v1/simulation/control")
def post_sim_control_v1(control: dict):
    global FLEET_SUMMARY_CACHE_SIGNATURE
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
            
    # Invalidate cache
    FLEET_SUMMARY_CACHE_SIGNATURE = ""
            
    return {
        "is_running": sim_manager.is_running,
        "speed": sim_manager.speed,
        "engines_active": len(engines_db)
    }

@app.get("/api/v1/research/benchmark")
async def get_benchmark_results_v1():
    """Triggers the cross-dataset transfer generalization benchmark suite off the event loop."""
    loop = asyncio.get_event_loop()
    def _run():
        return generate_benchmark_tables()
    try:
        results = await loop.run_in_executor(_thread_pool, _run)
        return {
            "status": "success",
            "latex": results["latex"],
            "markdown": results["markdown"],
            "results": results["results"],
            "p_value": results.get("p_value", 1.0),
            "reliability_data": results.get("reliability_data", {}),
            "pma_attributions": results.get("pma_attributions", {}),
            "faithfulness": results.get("faithfulness", {}),
            "baselines": results.get("baselines", {}),
            "ablation": results.get("ablation", {}),
            "seeds": results.get("seeds", []),
            "epochs": results.get("epochs", 0),
            "window_size": results.get("window_size", 30)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark execution failed: {e}")

@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    await sim_manager.connect(websocket)
    try:
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
            await websocket.receive_text()
    except WebSocketDisconnect:
        sim_manager.disconnect(websocket)
    except Exception:
        sim_manager.disconnect(websocket)

# Lifespan context manager for startup tasks
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app_instance):
    asyncio.create_task(sim_manager.run_loop())
    yield

app.router.lifespan_context = lifespan

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
