
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
# MIXUP & CUTMIX CONFIG
# ==========================================
CONFIG = {
    "data_dir": r"d:\personalProject\CSIRO-Image2Biomass_Prediction\csiro-biomass",
    "model_name": "convnext_small.fb_in22k_ft_in1k",
    "img_h": 224, 
    "img_w": 448,
    "batch_size": 16,
    "lr": 1e-4,
    "epochs": 50,
    "n_splits": 3,
    "mixup_prob": 0.4,
    "cutmix_prob": 0.4,
    "alpha": 1.0, # Beta distribution parameter
    "weights": {
        "biomass": 1.0,
        "meta_reg": 0.1,
        "meta_cls": 0.2,
        "consistency": 0.2
    },
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 4,
    "use_amp": True
}

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int64(W * cut_rat)
    cut_h = np.int64(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def apply_mixup_cutmix(images, bio_gt, reg_gt, cls_gt, num_species):
    """
    Applies Mixup or Cutmix to a batch.
    For classification (cls_gt), we convert to one-hot before mixing.
    """
    p = np.random.rand()
    if p > (CONFIG['mixup_prob'] + CONFIG['cutmix_prob']):
        return images, bio_gt, reg_gt, torch.nn.functional.one_hot(cls_gt, num_species).float(), 1.0, None

    batch_size = images.size(0)
    index = torch.randperm(batch_size).to(images.device)
    
    # Convert cls_gt to one-hot for mixing
    cls_onehot = torch.nn.functional.one_hot(cls_gt, num_species).float()
    
    lam = np.random.beta(CONFIG['alpha'], CONFIG['alpha'])
    
    if p < CONFIG['mixup_prob']:
        # Mixup
        images = lam * images + (1 - lam) * images[index, :]
    else:
        # Cutmix
        bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]
        # Adjust lambda to be proportional to area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
    
    # Mix labels
    bio_gt = lam * bio_gt + (1 - lam) * bio_gt[index, :]
    reg_gt = lam * reg_gt + (1 - lam) * reg_gt[index, :]
    cls_onehot = lam * cls_onehot + (1 - lam) * cls_onehot[index, :]
    
    return images, bio_gt, reg_gt, cls_onehot, lam, index

def train_mixup_cutmix():
    print(f"Starting MIXUP/CUTMIX Pipe on: {CONFIG['device']}")
    os.makedirs("models_checkpoints", exist_ok=True)

    df = pd.read_csv(os.path.join(CONFIG["data_dir"], "train.csv"))
    df_wide = df.pivot_table(index=['image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm'], 
                           columns='target_name', 
                           values='target').reset_index()
    
    num_species = len(df_wide['Species'].unique())
    gkf = GroupKFold(n_splits=CONFIG['n_splits'])
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df_wide, groups=df_wide['Sampling_Date'])):
        print(f"\n--- Fold {fold+1} (Mixup/Cutmix) ---")
        train_df, val_df = df_wide.iloc[train_idx], df_wide.iloc[val_idx]

        train_transform, val_transform = get_transforms(CONFIG['img_h'], CONFIG['img_w'])
        train_ds = AdvancedBiomassDataset(train_df, CONFIG['data_dir'], TARGET_COLUMNS, train_transform)
        val_ds = AdvancedBiomassDataset(val_df, CONFIG['data_dir'], TARGET_COLUMNS, val_transform)

        train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True)

        model = AdvancedHydraModel(model_name=CONFIG['model_name'], num_species=num_species).to(CONFIG['device'])
        model.load_state_dict(torch.load("models_checkpoints/best_advanced_ConvNextSmall_fold1.pth"))
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.05)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
        
        criterion_bio = WeightedHuberLoss(TARGET_WEIGHTS)
        criterion_cls = nn.BCEWithLogitsLoss() # Changed to BCE for mixed one-hot labels
        criterion_reg = nn.MSELoss()
        scaler = torch.amp.GradScaler('cuda', enabled=CONFIG['use_amp'])

        best_r2 = -float('inf')
        for epoch in range(CONFIG['epochs']):
            model.train()
            total_loss_accum = 0
            for images, bio_gt, reg_gt, cls_gt in train_loader:
                images = images.to(CONFIG['device'])
                bio_gt = bio_gt.to(CONFIG['device'])
                reg_gt = reg_gt.to(CONFIG['device'])
                cls_gt = cls_gt.to(CONFIG['device'])
                
                # Apply Augmentation
                images, bio_gt, reg_gt, cls_onehot, lam, _ = apply_mixup_cutmix(images, bio_gt, reg_gt, cls_gt, num_species)
                
                optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=CONFIG['use_amp']):
                    preds, p_reg, p_cls = model(images, return_meta=True)
                    
                    loss_bio = criterion_bio(preds, bio_gt) 
                    loss_reg = criterion_reg(p_reg, reg_gt)
                    loss_cls = criterion_cls(p_cls, cls_onehot)
                    
                    # Physical Consistency (Weighted by blended targets)
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

            # Validation (Standard, no Mixup)
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
                torch.save(model.state_dict(), f"models_checkpoints/best_mixup_ConvNextSmall_fold{fold+1}.pth")
    
if __name__ == "__main__":
    train_mixup_cutmix()
