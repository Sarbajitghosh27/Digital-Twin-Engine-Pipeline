import numpy as np
import torch

def compute_hi(err: float, p95_err: float) -> float:
    """Computes the Health Index based on percentile normalized reconstruction error."""
    return 100.0 * (1.0 - float(np.clip(err / (p95_err + 1e-9), 0.0, 1.0)))

def compute_hi_torch(err: torch.Tensor, p95_err: float) -> torch.Tensor:
    """Computes the Health Index using PyTorch tensors."""
    return 100.0 * (1.0 - torch.clamp(err / (p95_err + 1e-9), min=0.0, max=1.0))
