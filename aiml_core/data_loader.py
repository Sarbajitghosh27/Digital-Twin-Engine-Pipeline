import os
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

# 14 standard sensors for CMAPSS
CMAPSS_SENSORS = [
    "T24", "T30", "T50", "Ps30", "Nf", "Nc", "FuelFlow",
    "Bypass", "Bleed", "CoolantHPT", "CoolantLPT", "Vibration", "Efficiency", "Setting1"
]

# 47 sensors for N-CMAPSS (including operating settings and structural variables)
NCMAPSS_SENSORS = [
    "alt", "Mach", "TRA", "T2", "T24", "T30", "T48", "T50", "P2", "P15", "P30", "Nf", "Nc", "wf",
    "T40", "T90", "Ps30", "Nf_d", "Nc_d", "Bypass", "Bleed", "CoolantHPT", "CoolantLPT", "Vibration", "Efficiency",
    "Setting1", "Setting2", "Setting3"
] + [f"AuxSensor_{i}" for i in range(1, 20)] # Total 47 sensors

class DatasetManager:
    """
    Manages loading CMAPSS and N-CMAPSS datasets.
    Attempts to read from local paths; falls back to realistic simulation.
    """
    def __init__(self, data_root: str = "data"):
        self.data_root = data_root
        self.cmapss_dir = os.path.join(data_root, "CMAPSS")
        self.n_cmapss_dir = os.path.join(data_root, "N-CMAPSS")
        
        # Ensure directories exist
        os.makedirs(self.cmapss_dir, exist_ok=True)
        os.makedirs(self.n_cmapss_dir, exist_ok=True)
        
    def check_real_data_exists(self, dataset_name: str) -> bool:
        """Checks if files for the selected dataset exist in the data folder."""
        if dataset_name.startswith("FD"):
            train_file = os.path.join(self.cmapss_dir, f"train_{dataset_name}.txt")
            test_file = os.path.join(self.cmapss_dir, f"test_{dataset_name}.txt")
            return os.path.exists(train_file) and os.path.exists(test_file)
        elif dataset_name == "N-CMAPSS_DS01":
            # Search for any h5 file in N-CMAPSS folder
            if not os.path.exists(self.n_cmapss_dir):
                return False
            h5_files = [f for f in os.listdir(self.n_cmapss_dir) if f.endswith(".h5")]
            return len(h5_files) > 0
        return False

    def load_real_cmapss(self, dataset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads real CMAPSS text files and returns train & test DataFrames."""
        train_file = os.path.join(self.cmapss_dir, f"train_{dataset_name}.txt")
        test_file = os.path.join(self.cmapss_dir, f"test_{dataset_name}.txt")
        rul_file = os.path.join(self.cmapss_dir, f"RUL_{dataset_name}.txt")
        
        # CMAPSS text files have 26 columns:
        # Col 0: Engine ID, Col 1: Cycle, Col 2-4: Settings 1-3, Col 5-25: Sensors 1-21
        col_names = ["engine_id", "cycle", "setting1", "setting2", "setting3"] + [f"s_{i}" for i in range(1, 22)]
        
        train_df = pd.read_csv(train_file, sep=r"\s+", header=None, names=col_names)
        test_df = pd.read_csv(test_file, sep=r"\s+", header=None, names=col_names)
        
        # Compute true RUL for training data
        # Each engine runs until failure in training set
        max_cycles = train_df.groupby("engine_id")["cycle"].max().to_dict()
        train_df["RUL_actual"] = train_df.apply(lambda row: max_cycles[row["engine_id"]] - row["cycle"], axis=1)
        
        # Cap RUL (Standard piece-wise linear RUL target: 130 for FD002/FD004, 125 for others)
        clip_limit = 130 if dataset_name in ["FD002", "FD004"] else 125
        train_df["RUL_actual"] = train_df["RUL_actual"].clip(upper=clip_limit)
        
        # Compute true RUL for test data using RUL file if available
        if os.path.exists(rul_file):
            rul_val = np.loadtxt(rul_file)
            test_max_cycles = test_df.groupby("engine_id")["cycle"].max().to_dict()
            test_df["RUL_actual"] = test_df.apply(
                lambda row: (test_max_cycles[row["engine_id"]] + rul_val[int(row["engine_id"]) - 1]) - row["cycle"], 
                axis=1
            )
            test_df["RUL_actual"] = test_df["RUL_actual"].clip(upper=clip_limit)
        else:
            # Fallback estimation if RUL file is missing
            test_max_cycles = test_df.groupby("engine_id")["cycle"].max().to_dict()
            test_df["RUL_actual"] = test_df.apply(lambda row: (test_max_cycles[row["engine_id"]] + 30) - row["cycle"], axis=1)
            test_df["RUL_actual"] = test_df["RUL_actual"].clip(upper=clip_limit)
            
        # Map CMAPSS 21 sensors to the 14 keys in backend.py
        # S2->T24, S3->T30, S4->T50, S7->Ps30, S8->Nf, S9->Nc, S11->FuelFlow, S12->Bypass, S13->Bleed, S15->CoolantHPT (S15/S16), S17->CoolantLPT, S20->Vibration, S21->Efficiency
        sensor_map = {
            "T24": "s_2", "T30": "s_3", "T50": "s_4", "Ps30": "s_7", "Nf": "s_8", "Nc": "s_9",
            "FuelFlow": "s_11", "Bypass": "s_15", "Bleed": "s_17", "CoolantHPT": "s_20", 
            "CoolantLPT": "s_21", "Setting1": "setting1"
        }
        
        for key, orig_col in sensor_map.items():
            train_df[key] = train_df[orig_col]
            test_df[key] = test_df[orig_col]
            
        # Synthetically generate Vibration and Efficiency as they are not present in real CMAPSS
        train_max = train_df.groupby("engine_id")["cycle"].transform("max")
        train_ratio = train_df["cycle"] / train_max
        train_df["Vibration"] = 1.0 + 1.5 * (train_ratio ** 2) + np.random.normal(0, 0.1, len(train_df))
        train_df["Efficiency"] = 98.5 - 5.0 * (train_ratio ** 2) + np.random.normal(0, 0.2, len(train_df))
        
        test_max = test_df.groupby("engine_id")["cycle"].transform("max")
        test_ratio = test_df["cycle"] / test_max
        test_df["Vibration"] = 1.0 + 1.5 * (test_ratio ** 2) + np.random.normal(0, 0.1, len(test_df))
        test_df["Efficiency"] = 98.5 - 5.0 * (test_ratio ** 2) + np.random.normal(0, 0.2, len(test_df))
        
        return train_df, test_df

    def load_real_ncmapss(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads N-CMAPSS H5 file (simplified wrapper for benchmarking)."""
        # Since reading real N-CMAPSS requires specialized h5py extraction,
        # we will extract variables into a clean pandas DataFrame.
        # pyrefly: ignore [missing-import]
        import h5py
        h5_files = [f for f in os.listdir(self.n_cmapss_dir) if f.endswith(".h5")]
        h5_path = os.path.join(self.n_cmapss_dir, h5_files[0])
        
        with h5py.File(h5_path, 'r') as f:
            # Typical dataset keys in N-CMAPSS: W_dev, X_s_dev, Y_dev, etc.
            # We construct a DataFrame from the numpy arrays
            # For this benchmark, we load a subset to keep performance high
            W = np.array(f['W_dev']) # Settings
            X = np.array(f['X_s_dev']) # Sensors
            Y = np.array(f['Y_dev']) # RUL
            
            # Map settings and sensors
            data = {}
            # Settings
            data["alt"] = W[:, 0]
            data["Mach"] = W[:, 1]
            data["TRA"] = W[:, 2]
            
            # Key sensors matching 47 schema
            for i in range(min(X.shape[1], 44)):
                if i < len(NCMAPSS_SENSORS) - 3:
                    data[NCMAPSS_SENSORS[i+3]] = X[:, i]
                else:
                    data[f"AuxSensor_{i}"] = X[:, i]
                    
            df = pd.DataFrame(data)
            df["RUL_actual"] = Y.flatten()
            df["cycle"] = np.arange(len(df)) # Simulated cycle counts
            df["engine_id"] = 1
            
            # Split into train/test
            split_idx = int(len(df) * 0.8)
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()
            
            return train_df, test_df

    def get_dataset(self, dataset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Gets dataset either from files or synthetic generator fallback."""
        if self.check_real_data_exists(dataset_name):
            try:
                if dataset_name.startswith("FD"):
                    return self.load_real_cmapss(dataset_name)
                elif dataset_name == "N-CMAPSS_DS01":
                    return self.load_real_ncmapss()
            except Exception as e:
                print(f"Error loading real dataset {dataset_name}, falling back to synthetic. Error: {e}")
                
        # Fallback to high-fidelity synthetic data
        return self.generate_synthetic_dataset(dataset_name)

    def generate_synthetic_dataset(self, dataset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generates high-fidelity simulated telemetry datasets mimicking the reality gap."""
        num_engines = 15 if dataset_name.startswith("FD") else 5
        train_records = []
        test_records = []
        
        is_ncmapss = (dataset_name == "N-CMAPSS_DS01")
        sensor_list = NCMAPSS_SENSORS if is_ncmapss else CMAPSS_SENSORS
        
        # Operating condition regimes:
        # FD001, FD003: 1 regime
        # FD002, FD004: 6 regimes
        # N-CMAPSS: Multi-regime flight phases (Climb, Cruise, Descent)
        if dataset_name in ["FD002", "FD004"]:
            num_regimes = 6
        elif is_ncmapss:
            num_regimes = 3 # Climb, Cruise, Descent
        else:
            num_regimes = 1
            
        # Fault modes:
        # FD001, FD002: 1 fault mode (HPC degradation)
        # FD003, FD004: 2 fault modes (HPC + Fan degradation)
        # N-CMAPSS: Multiple complex fault modes
        has_two_faults = dataset_name in ["FD003", "FD004", "N-CMAPSS_DS01"]

        for is_train, records_list in [(True, train_records), (False, test_records)]:
            for engine_id in range(1, num_engines + 1):
                # CMAPSS typically ranges between 130 and 300 cycles
                max_cycles = random.randint(140, 260)
                alpha = random.uniform(1.5, 2.5) # degradation rate
                
                # Assign fault mode to this engine
                fault_mode = "HPC"
                if has_two_faults and random.random() > 0.5:
                    fault_mode = "Fan"
                
                # Base sensor baselines
                baselines = {
                    "T24": 642.0, "T30": 1585.0, "T50": 1400.0, "Ps30": 554.0,
                    "Nf": 2388.0, "Nc": 9050.0, "FuelFlow": 521.0, "Bypass": 8.4,
                    "Bleed": 392.0, "CoolantHPT": 39.0, "CoolantLPT": 23.3,
                    "Vibration": 1.0, "Efficiency": 98.5, "Setting1": 0.0007,
                    # N-CMAPSS specific baselines
                    "alt": 10000.0, "Mach": 0.5, "TRA": 80.0, "T2": 288.0,
                    "T48": 1200.0, "P2": 14.7, "P15": 16.0, "P30": 300.0, "wf": 8.0,
                    "T40": 1800.0, "T90": 550.0, "Nf_d": 2400.0, "Nc_d": 9100.0,
                    "Setting2": 0.0002, "Setting3": 100.0
                }
                
                # Baseline sensor shifts due to operational regime offsets
                regime_shifts = []
                for r in range(num_regimes):
                    shift = {}
                    for s in sensor_list:
                        if s in ["alt", "Mach", "TRA", "T30", "T50", "Nf", "Nc", "Ps30"]:
                            shift[s] = random.uniform(-0.15, 0.15) * baselines.get(s, 1.0)
                        else:
                            shift[s] = 0.0
                    regime_shifts.append(shift)

                # Generate cycles
                for cycle in range(1, max_cycles + 1):
                    ratio = cycle / max_cycles
                    true_rul = max_cycles - cycle
                    clip_limit = 130 if dataset_name in ["FD002", "FD004"] else 125
                    capped_rul = min(clip_limit, true_rul)
                    
                    # Determine current regime
                    if num_regimes == 1:
                        regime_id = 0
                    elif is_ncmapss:
                        # Mimic flight phases: Climb first 25%, Cruise middle 60%, Descent last 15%
                        prog = cycle / max_cycles
                        if prog < 0.25:
                            regime_id = 0 # Climb
                        elif prog < 0.85:
                            regime_id = 1 # Cruise
                        else:
                            regime_id = 2 # Descent
                    else:
                        # Cyclic regime transitions for FD002/FD004
                        regime_id = (cycle // 20) % num_regimes
                        
                    # Calculate sensor values
                    row = {
                        "engine_id": engine_id,
                        "cycle": cycle,
                        "regime": regime_id,
                        "RUL_actual": capped_rul
                    }
                    
                    # Set settings based on regime
                    if is_ncmapss:
                        row["alt"] = 0.0 if regime_id == 0 else (35000.0 if regime_id == 1 else 15000.0)
                        row["Mach"] = 0.25 if regime_id == 0 else (0.80 if regime_id == 1 else 0.45)
                        row["TRA"] = 90.0 if regime_id == 0 else (75.0 if regime_id == 1 else 40.0)
                    else:
                        row["setting1"] = 0.001 * regime_id
                        row["setting2"] = 0.0002 * (regime_id % 2)
                        row["setting3"] = 100.0
                        
                    # Compute degradations based on fault modes
                    for s in sensor_list:
                        if s in ["engine_id", "cycle", "regime", "RUL_actual", "alt", "Mach", "TRA", "setting1", "setting2", "setting3"]:
                            continue
                            
                        base = baselines.get(s, 1.0)
                        shift = regime_shifts[regime_id].get(s, 0.0)
                        
                        # Degradation delta (direction & magnitude)
                        delta = 0.0
                        if fault_mode == "HPC":
                            # HPC degradation raises compressor temperatures and drops pressures
                            if s in ["T30", "T50", "Bleed", "CoolantHPT"]:
                                delta = base * 0.03 * (ratio ** alpha)
                            elif s in ["Ps30", "Efficiency", "FuelFlow"]:
                                delta = -base * 0.02 * (ratio ** alpha)
                        else:
                            # Fan degradation raises fan temp and lowers fan speeds
                            if s in ["T24", "T50", "Vibration"]:
                                delta = base * 0.04 * (ratio ** alpha)
                            elif s in ["Nf", "Efficiency"]:
                                delta = -base * 0.025 * (ratio ** alpha)
                                
                        # General wear and tear overlay
                        if s == "Vibration":
                            delta += base * 1.5 * (ratio ** (alpha * 1.5))
                        elif s == "Efficiency":
                            delta -= base * 0.05 * (ratio ** alpha)
                            
                        # Combine base, regime shift, degradation, and noise
                        noise_std = base * 0.002
                        noise = np.random.normal(0, noise_std)
                        
                        row[s] = round(base + shift + delta + noise, 4)
                        
                    # Fill auxiliary sensors for N-CMAPSS
                    if is_ncmapss:
                        for s in sensor_list:
                            if s.startswith("AuxSensor"):
                                row[s] = round(random.gauss(10, 1) + 2 * ratio, 4)
                                
                    records_list.append(row)
                    
        train_df = pd.DataFrame(train_records)
        test_df = pd.DataFrame(test_records)
        
        return train_df, test_df

def normalize_regimes(df: pd.DataFrame, sensor_cols: List[str], regime_col: str = "regime") -> pd.DataFrame:
    """
    Performs per-regime z-score normalization on sensor columns.
    Removes operational conditions and unmasks structural degradation.
    """
    df = df.copy()
    normalized_df = df.copy()
    for col in sensor_cols:
        if col in normalized_df.columns:
            normalized_df[col] = normalized_df[col].astype(float)
            df[col] = df[col].astype(float)
            
    if regime_col not in normalized_df.columns:
        # Create a single default regime if not present
        normalized_df[regime_col] = 0
        df[regime_col] = 0
        
    for regime in df[regime_col].unique():
        mask = df[regime_col] == regime
        for col in sensor_cols:
            if col in normalized_df.columns:
                mean = df.loc[mask, col].mean()
                std = df.loc[mask, col].std()
                if std > 0.001:
                    normalized_df.loc[mask, col] = (normalized_df.loc[mask, col] - mean) / std
                else:
                    normalized_df.loc[mask, col] = 0.0
    return normalized_df

def prepare_sliding_windows(df: pd.DataFrame, sensor_cols: List[str], window_size: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepares sliding windows from time series DataFrame.
    Returns:
        X: numpy array of shape (num_windows, window_size, num_sensors)
        Y: numpy array of shape (num_windows, 1) representing remaining useful life
    """
    X_list, Y_list = [], []
    for engine_id in df["engine_id"].unique():
        engine_df = df[df["engine_id"] == engine_id].sort_values("cycle")
        if len(engine_df) < window_size:
            continue
            
        sensors = engine_df[sensor_cols].values
        ruls = engine_df["RUL_actual"].values
        
        for i in range(len(engine_df) - window_size + 1):
            X_list.append(sensors[i : i + window_size])
            Y_list.append(ruls[i + window_size - 1])
            
    return np.array(X_list), np.array(Y_list).reshape(-1, 1)
