
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score

# ==========================================
# COMPETITION TARGET WEIGHTS
# ==========================================
TARGET_COLUMNS = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'GDM_g', 'Dry_Total_g']
TARGET_WEIGHTS = [0.1, 0.1, 0.1, 0.2, 0.5]

class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
        
    def forward(self, inputs, targets):
        loss = (inputs - targets)**2
        weighted_loss = loss * self.weights.to(inputs.device)
        return weighted_loss.mean()

class WeightedHuberLoss(nn.Module):
    def __init__(self, weights, delta=1.0):
        super().__init__()
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
        self.delta = delta
        
    def forward(self, inputs, targets):
        # Huber Loss: quadratic for small errors, linear for large errors
        abs_diff = torch.abs(inputs - targets)
        quadratic = torch.clamp(abs_diff, max=self.delta)
        linear = abs_diff - quadratic
        loss = 0.5 * quadratic**2 + self.delta * linear
        
        weighted_loss = loss * self.weights.to(inputs.device)
        return weighted_loss.mean()

def competition_metric(y_true, y_pred):
    """
    Globally weighted coefficient of determination (R2)
    calculated across all image and target pairs.
    Formula: R2_w = 1 - [sum w_j(y_j - y_hat_j)^2] / [sum w_j(y_j - y_avg_w)^2]
    """
    # Flatten everything to handle "across all image and target pairs"
    # y_true, y_pred are [N, 5]
    N = y_true.shape[0]
    weights = np.array(TARGET_WEIGHTS)
    
    # Broadcast weights to match [N, 5]
    w = np.tile(weights, (N, 1)) # [N, 5]
    
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    w_flat = w.flatten()
    
    # 1. Weighted Mean of Ground Truth
    y_avg_w = np.sum(w_flat * y_true_flat) / np.sum(w_flat)
    
    # 2. Weighted Sum of Squares Residuals (Numerator)
    ss_res = np.sum(w_flat * (y_true_flat - y_pred_flat)**2)
    
    # 3. Weighted total Sum of Squares (Denominator)
    ss_tot = np.sum(w_flat * (y_true_flat - y_avg_w)**2)
    
    # 4. Final R2
    if ss_tot == 0:
        return 0.0
        
    return 1 - (ss_res / ss_tot)
