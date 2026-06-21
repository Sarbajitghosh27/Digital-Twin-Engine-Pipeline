# Centralized configuration for AI/ML models and training hyperparameters
import os

CONFIG = {
    "window_size": 30,
    "ae_hidden_dim": 8,
    "lstm_hidden_dim": 16,
    "learning_rate": 0.01,
    "epochs": 20,
    "batch_size": 64,
    "validation_split": 0.15,
    "early_stopping_patience": 4,
    "seeds": [42, 123, 7],
    "checkpoints_dir": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
}
