
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt

from data.dataset import BiomassDataset, get_transforms
from models.hydra import HydraVisionModel
from utils.metrics import WeightedHuberLoss, TARGET_COLUMNS, TARGET_WEIGHTS, competition_metric

# ==========================================
# HYDRA VISION CONFIG
# ==========================================
CONFIG = {
    "data_dir": r"d:\personalProject\CSIRO-Image2Biomass_Prediction\csiro-biomass",
    "model_name": "convnext_tiny.in12k_ft_in1k", 
    "img_h": 224, 
    "img_w": 448,
    "batch_size": 12,
    "lr": 1e-4,
    "epochs": 40,
    "n_splits": 5,
    "consistency_weight": 0.3, # Loss weight for Total = Sum(Parts)
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 0
}

def train_hydra_vision():
    print(f"Starting Hydra-Vision Training on: {CONFIG['device']}")
    os.makedirs("models_checkpoints", exist_ok=True)

    # 1. LOAD & PIVOT DATA
    df = pd.read_csv(os.path.join(CONFIG["data_dir"], "train.csv"))
    
    # Simple pivot to get wide format (one row per image/date)
    # Note: We keep Sampling_Date for GroupKFold
    df_wide = df.pivot_table(index=['image_path', 'Sampling_Date'], 
                           columns='target_name', 
                           values='target').reset_index()
    df_wide.columns.name = None
    
    # 2. CROSS-VALIDATION SPLIT (Grouped by Sampling_Date)
    gkf = GroupKFold(n_splits=CONFIG['n_splits'])
    
    # Column indices for consistency check
    # clover=0, dead=1, green=2, gdm=3, total=4
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df_wide, groups=df_wide['Sampling_Date'])):
        print(f"\n--- Training Fold {fold+1} ---")
        train_df = df_wide.iloc[train_idx]
        val_df = df_wide.iloc[val_idx]

        # 3. DATA LOADERS (Empty tabular columns as we are Vision-Only)
        train_transform, val_transform = get_transforms(CONFIG['img_h'], CONFIG['img_w'])
        train_ds = BiomassDataset(train_df, CONFIG['data_dir'], TARGET_COLUMNS, [], transform=train_transform)
        val_ds = BiomassDataset(val_df, CONFIG['data_dir'], TARGET_COLUMNS, [], transform=val_transform)

        train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'])
        val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])

        # 4. MODEL (Hydra: 5 independent heads)
        model = HydraVisionModel(model_name=CONFIG['model_name']).to(CONFIG['device'])
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=2e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
        
        criterion = WeightedHuberLoss(TARGET_WEIGHTS)
        
        def consistency_loss(outputs):
            parts_sum = outputs[:, 0] + outputs[:, 1] + outputs[:, 2] # Clover + Dead + Green
            total_pred = outputs[:, 4] # Total
            return torch.mean((parts_sum - total_pred)**2)

        # 5. TRAINING LOOP
        best_r2 = -float('inf')
        for epoch in range(CONFIG['epochs']):
            model.train()
            total_loss_accum = 0
            for images, _, targets in train_loader:
                images, targets = images.to(CONFIG['device']), targets.to(CONFIG['device'])
                
                optimizer.zero_grad()
                outputs = model(images)
                
                # Combined Loss: Huber Loss + Physical Consistency Loss
                main_loss = criterion(outputs, targets)
                consist_loss = consistency_loss(outputs)
                loss = main_loss + CONFIG['consistency_weight'] * consist_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss_accum += loss.item()
            
            scheduler.step()

            # Validation
            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for images, _, targets in val_loader:
                    images = images.to(CONFIG['device'])
                    outputs = model(images)
                    all_preds.append(outputs.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())

            r2 = competition_metric(np.vstack(all_targets), np.vstack(all_preds))
            print(f"Epoch {epoch+1} | Total Loss: {total_loss_accum/len(train_loader):.4f} | R2: {r2:.4f}")

            if r2 > best_r2:
                best_r2 = r2
                torch.save(model.state_dict(), f"models_checkpoints/best_hydra_vision_fold{fold+1}.pth")
        
        # For quick local testing, we can break after Fold 1
        # break 

if __name__ == "__main__":
    train_hydra_vision()
