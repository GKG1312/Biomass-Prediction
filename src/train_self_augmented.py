
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt

from data.dataset import get_transforms
from data.self_augmented_dataset import SelfAugmentedBiomassDataset
from models.self_augmented import SelfAugmentedHydraModel
from utils.metrics import WeightedHuberLoss, TARGET_COLUMNS, TARGET_WEIGHTS, competition_metric

# ==========================================
# SELF-AUGMENTED CONFIG
# ==========================================
CONFIG = {
    "data_dir": r"d:\personalProject\CSIRO-Image2Biomass_Prediction\csiro-biomass",
    "model_name": "convnext_base.clip_laion2b_augreg_ft_in12k",   #"convnext_tiny.in12k_ft_in1k", 
    "img_h": 384, 
    "img_w": 384,
    "batch_size": 12,
    "lr": 1e-4,
    "epochs": 50,
    "n_splits": 3,
    "weights": {
        "biomass": 1.0,
        "meta": 0.2,       # Auxiliary loss weight for [NDVI, Height, Species]
        "consistency": 0.3
    },
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 4, 
    "pin_memory": True,
    "use_amp": True 
}

def train():
    if CONFIG['device'] == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    print(f"Starting Optimized Self-Augmented Pipeline on: {CONFIG['device']}")
    os.makedirs("models_checkpoints", exist_ok=True)

    # 1. LOAD DATA
    df = pd.read_csv(os.path.join(CONFIG["data_dir"], "train.csv"))
    df_wide = df.pivot_table(index=['image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm'], 
                           columns='target_name', 
                           values='target').reset_index()
    df_wide.columns.name = None
    
    # 2. CROSS-VALIDATION
    gkf = GroupKFold(n_splits=CONFIG['n_splits'])
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df_wide, groups=df_wide['Sampling_Date'])):
        print(f"\n--- Training Fold {fold+1} ---")
        train_df = df_wide.iloc[train_idx]
        val_df = df_wide.iloc[val_idx]

        # 3. DATA LOADERS
        train_transform, val_transform = get_transforms(CONFIG['img_h'], CONFIG['img_w'])
        train_ds = SelfAugmentedBiomassDataset(train_df, CONFIG['data_dir'], TARGET_COLUMNS, train_transform)
        val_ds = SelfAugmentedBiomassDataset(val_df, CONFIG['data_dir'], TARGET_COLUMNS, val_transform)

        train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, 
                                  num_workers=CONFIG['num_workers'], pin_memory=CONFIG['pin_memory'])
        val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, 
                                num_workers=CONFIG['num_workers'], pin_memory=CONFIG['pin_memory'])

        # 4. MODEL & LOSS
        model = SelfAugmentedHydraModel(model_name=CONFIG['model_name']).to(CONFIG['device'])
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=2e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
        
        criterion_biomass = WeightedHuberLoss(TARGET_WEIGHTS)
        criterion_meta = nn.MSELoss() # MSE for all 3 meta values
        
        def consistency_loss(outputs):
            parts_sum = outputs[:, 0] + outputs[:, 1] + outputs[:, 2] 
            total_pred = outputs[:, 4] 
            return torch.mean((parts_sum - total_pred)**2)

        # 5. TRAINING LOOP
        scaler = torch.amp.GradScaler('cuda', enabled=CONFIG['use_amp'])
        best_r2 = -float('inf')
        
        for epoch in range(CONFIG['epochs']):
            model.train()
            total_loss_item = 0
            for images, biomass_gt, meta_gt in train_loader:
                images, biomass_gt, meta_gt = images.to(CONFIG['device']), biomass_gt.to(CONFIG['device']), meta_gt.to(CONFIG['device'])
                
                optimizer.zero_grad()
                
                with torch.amp.autocast('cuda', enabled=CONFIG['use_amp']):
                    preds, pred_meta = model(images, return_meta=True)
                    
                    # Losses
                    loss_bio = criterion_biomass(preds, biomass_gt)
                    loss_meta = criterion_meta(pred_meta, meta_gt)
                    loss_cons = consistency_loss(preds)
                    
                    loss_total = (CONFIG['weights']['biomass'] * loss_bio) + \
                                 (CONFIG['weights']['meta'] * loss_meta) + \
                                 (CONFIG['weights']['consistency'] * loss_cons)
                
                scaler.scale(loss_total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                total_loss_item += loss_total.item()
            
            scheduler.step()

            # Validation
            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for images, biomass_gt, _ in val_loader:
                    images = images.to(CONFIG['device'])
                    out_bio = model(images)
                    all_preds.append(out_bio.cpu().numpy())
                    all_targets.append(biomass_gt.cpu().numpy())

            r2 = competition_metric(np.vstack(all_targets), np.vstack(all_preds))
            print(f"Epoch {epoch+1} | Loss: {total_loss_item/len(train_loader):.4f} | R2: {r2:.4f}")

            if r2 > best_r2:
                best_r2 = r2
                torch.save(model.state_dict(), f"models_checkpoints/best_self_augmented_fold{fold+1}.pth")
        
        # break

if __name__ == "__main__":
    train()
