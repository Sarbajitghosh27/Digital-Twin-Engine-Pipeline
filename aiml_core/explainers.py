import numpy as np
from typing import Callable

class PMAExplainer:
    """
    Perturbation Marginal Attribution (PMA) Explainer.
    A perturbation-based local feature attribution method designed for sensor timeseries
    and tabular predictions. Computes attributions by replacing each feature with its
    healthy baseline value and scaling the marginal changes to sum exactly to the prediction
    difference.
    
    Rather than relying on exact Shapley axiom approximations, the validity of this 
    method is verified empirically via Area Under the Deletion Curve (AUDC) comparison.
    """
    def __init__(self, model_func: Callable[[np.ndarray], float], baseline_state: np.ndarray):
        """
        Args:
            model_func: A callable function that takes a numpy array input of shape
                        (batch_size, seq_len, num_features) or (batch_size, num_features)
                        and returns a scalar prediction.
            baseline_state: NumPy array representing the healthy reference baseline.
                            Shape should match the input dimensions.
        """
        self.model_func = model_func
        self.baseline_state = np.copy(baseline_state)
        self.input_dim = baseline_state.shape[-1]
        
    def explain(self, current_state: np.ndarray) -> np.ndarray:
        """
        Computes attributions for each feature.
        
        Args:
            current_state: NumPy array of the current state to explain.
                           Shape: (1, seq_len, num_features) or (1, num_features)
        Returns:
            attributions: NumPy array of shape (num_features,) representing feature contributions.
        """
        # 1. Compute current and baseline predictions
        y_curr = self.model_func(current_state)
        y_base = self.model_func(self.baseline_state)
        
        # Ensure we are dealing with scalar floats
        if hasattr(y_curr, "__len__") or isinstance(y_curr, np.ndarray):
            y_curr = float(y_curr[0])
        if hasattr(y_base, "__len__") or isinstance(y_base, np.ndarray):
            y_base = float(y_base[0])
            
        delta = y_curr - y_base
        
        if abs(delta) < 1e-6:
            # If there's no change, all attributions are zero
            return np.zeros(self.input_dim)
            
        # 2. Compute marginal change for each feature
        marginals = np.zeros(self.input_dim)
        is_3d = len(current_state.shape) == 3
        
        for i in range(self.input_dim):
            # Create a perturbed state where feature i is replaced by baseline
            perturbed = np.copy(current_state)
            if is_3d:
                # Replace the entire sequence of feature i
                perturbed[0, :, i] = self.baseline_state[0, :, i]
            else:
                # Replace feature i
                perturbed[0, i] = self.baseline_state[0, i]
                
            y_perturbed = self.model_func(perturbed)
            if hasattr(y_perturbed, "__len__") or isinstance(y_perturbed, np.ndarray):
                y_perturbed = float(y_perturbed[0])
            
            # The marginal contribution is the change when restoring feature i to current
            # d_i = y_curr - y_perturbed
            marginals[i] = y_curr - y_perturbed
            
        # 3. Scale marginals to sum exactly to delta (additivity)
        marginals_sum = np.sum(marginals)
        if abs(marginals_sum) > 1e-6:
            scaling_factor = delta / marginals_sum
            attributions = marginals * scaling_factor
        else:
            # If marginals sum to zero (e.g. no feature had impact), distribute delta evenly
            attributions = np.ones(self.input_dim) * (delta / self.input_dim)
            
        return attributions
