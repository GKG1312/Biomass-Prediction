
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold

from data.dataset import get_transforms
from data.advanced_dataset import AdvancedBiomassDataset
from models.advanced_hydra import AdvancedHydraModel
from utils.metrics import WeightedHuberLoss, TARGET_COLUMNS, TARGET_WEIGHTS, competition_metric

# ==========================================
# ADVANCED TRAINING CONFIG
# ==========================================
CONFIG = {
    "data_dir": r"d:\personalProject\CSIRO-Image2Biomass_Prediction\csiro-biomass",
    "model_name": "convnext_small.fb_in22k_ft_in1k",  #"convnext_base.clip_laion2b_augreg_ft_in12k", 
    "img_h": 224, 
    "img_w": 448,
    "batch_size": 12,
    "lr": 1e-4,
    "epochs": 50,
    "n_splits": 3,
    "weights": {
        "biomass": 1.0,
        "meta_reg": 0.1,    # [NDVI, Height]
        "meta_cls": 0.2,    # Species CrossEntropy
        "consistency": 0.2  # Clover + Green + Dead = Total
    },
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 4,
    "use_amp": True
}

def log_cosh_loss(y_pred, y_true):
    return torch.mean(torch.log(torch.cosh(y_pred - y_true)))

def train_advanced():
    print(f"Starting ADVANCED Pipeline on: {CONFIG['device']}")
    os.makedirs("models_checkpoints", exist_ok=True)

    df = pd.read_csv(os.path.join(CONFIG["data_dir"], "train.csv"))
    df_wide = df.pivot_table(index=['image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm'], 
                           columns='target_name', 
                           values='target').reset_index()
    
    num_species = len(df_wide['Species'].unique())
    gkf = GroupKFold(n_splits=CONFIG['n_splits'])
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df_wide, groups=df_wide['Sampling_Date'])):
        print(f"\n--- Training Fold {fold+1} ---")
        train_df, val_df = df_wide.iloc[train_idx], df_wide.iloc[val_idx]

        train_transform, val_transform = get_transforms(CONFIG['img_h'], CONFIG['img_w'])
        train_ds = AdvancedBiomassDataset(train_df, CONFIG['data_dir'], TARGET_COLUMNS, train_transform)
        val_ds = AdvancedBiomassDataset(val_df, CONFIG['data_dir'], TARGET_COLUMNS, val_transform)

        train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True)

        model = AdvancedHydraModel(model_name=CONFIG['model_name'], num_species=num_species).to(CONFIG['device'])
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.05)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
        
        criterion_bio = WeightedHuberLoss(TARGET_WEIGHTS)
        criterion_cls = nn.CrossEntropyLoss()
        criterion_reg = nn.MSELoss()
        scaler = torch.amp.GradScaler('cuda', enabled=CONFIG['use_amp'])

        best_r2 = -float('inf')
        for epoch in range(CONFIG['epochs']):
            model.train()
            total_loss_accum = 0
            for images, bio_gt, reg_gt, cls_gt in train_loader:
                images, bio_gt = images.to(CONFIG['device']), bio_gt.to(CONFIG['device'])
                reg_gt, cls_gt = reg_gt.to(CONFIG['device']), cls_gt.to(CONFIG['device'])
                
                optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=CONFIG['use_amp']):
                    preds, p_reg, p_cls = model(images, return_meta=True)
                    
                    # Numeric Safety: Huber is robust, log_cosh is dangerous at this scale
                    loss_bio = criterion_bio(preds, bio_gt) 
                    loss_reg = criterion_reg(p_reg, reg_gt)
                    loss_cls = criterion_cls(p_cls, cls_gt)
                    
                    # Physical Consistency
                    loss_cons = torch.mean((preds[:,:3].sum(1) - preds[:,4])**2)
                    
                    loss = (CONFIG['weights']['biomass'] * loss_bio) + \
                           (CONFIG['weights']['meta_reg'] * loss_reg) + \
                           (CONFIG['weights']['meta_cls'] * loss_cls) + \
                           (CONFIG['weights']['consistency'] * loss_cons)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                total_loss_accum += loss.item()

            scheduler.step()

            # Validation
            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for images, bio_gt, _, _ in val_loader:
                    out = model(images.to(CONFIG['device']))
                    all_preds.append(out.cpu().numpy())
                    all_targets.append(bio_gt.numpy())

            r2 = competition_metric(np.vstack(all_targets), np.vstack(all_preds))
            print(f"Epoch {epoch+1} | Loss: {total_loss_accum/len(train_loader):.4f} | R2: {r2:.4f}")

            if r2 > best_r2:
                best_r2 = r2
                torch.save(model.state_dict(), f"models_checkpoints/best_advanced_ConvNextSmall_fold{fold+1}.pth")
        
        # break

if __name__ == "__main__":
    train_advanced()
