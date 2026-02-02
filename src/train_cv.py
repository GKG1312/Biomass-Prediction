
import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import json
import matplotlib.pyplot as plt

from data.dataset import BiomassDataset, get_transforms
from models.baseline import MultiModalConvNeXt
from models.contextual import EnvironmentalContextModel
from utils.metrics import WeightedHuberLoss, TARGET_COLUMNS, TARGET_WEIGHTS, competition_metric

# ==========================================
# LOCAL OPTIMIZED CONFIG
# ==========================================
CONFIG = {
    # Update this path to your local data location
    "data_dir": r"d:\personalProject\CSIRO-Image2Biomass_Prediction\csiro-biomass",
    
    # Using 'convnext_tiny' for local speed. Switch back to 'base' if you have a powerful GPU.
    "model_name": "convnext_nano.in12k_ft_in1k", 

    "resume": True,
    
    # 2:1 Aspect Ratio (Low res for local)
    "img_h": 224,
    "img_w": 448,
    
    "batch_size": 16, 
    "lr": 5e-4,
    "epochs": 50,
    "val_size": 0.2, # 20% for validation
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 0 # Set to 0 for Windows local stability
}

def train_local():
    os.makedirs("models_checkpoints", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    print(f"Starting Local Training on: {CONFIG['device']}")
    
    # 1. LOAD DATA
    df = pd.read_csv(os.path.join(CONFIG['data_dir'], "train.csv"))
    
    # Pre-process Temporal & Categorical
    df['Sampling_Date'] = pd.to_datetime(df['Sampling_Date'])
    
    # CYCLICAL DATE ENCODING
    df['DayOfYear'] = df['Sampling_Date'].dt.dayofyear
    df['sin_day'] = np.sin(2 * np.pi * df['DayOfYear'] / 365.0)
    df['cos_day'] = np.cos(2 * np.pi * df['DayOfYear'] / 365.0)
    
    # CATEGORICAL ENCODING
    # We use get_dummies for State and Species
    df = pd.get_dummies(df, columns=['State', 'Species'], prefix=['State', 'Sp'])
    
    # Identify all categorical encoded columns + numerical cols
    encoded_cols = [c for c in df.columns if c.startswith('State_') or c.startswith('Sp_')]
    # tab_cols = ['Pre_GSHH_NDVI', 'Height_Ave_cm', 'DayOfYear'] + encoded_cols
    # Using sin_day/cos_day instead of raw DayOfYear
    tab_cols = ['Pre_GSHH_NDVI', 'Height_Ave_cm', 'sin_day', 'cos_day'] + encoded_cols
    
    # Pivot to wide format for image-level cross validation
    # Include all metadata columns in the index to preserve them
    df_wide = df.pivot_table(index=['image_path'] + tab_cols, 
                           columns='target_name', 
                           values='target').reset_index()
    
    # 2. SIMPLE TRAIN/VAL SPLIT
    unique_images = df_wide['image_path'].unique()
    train_imgs, val_imgs = train_test_split(unique_images, test_size=CONFIG['val_size'], random_state=42)
    
    train_df = df_wide[df_wide['image_path'].isin(train_imgs)].copy()
    val_df = df_wide[df_wide['image_path'].isin(val_imgs)].copy()
    
    # 3. SCALING
    scaler = StandardScaler()
    train_df[tab_cols] = scaler.fit_transform(train_df[tab_cols])
    val_df[tab_cols] = scaler.transform(val_df[tab_cols])
    
    # Save the scaler and the list of columns to ensure consistency in inference
    metadata_info = {
        'tab_cols': tab_cols,
        'scaler': scaler
    }
    joblib.dump(metadata_info, "models_checkpoints/metadata_info_local_contextual.pkl")
    
    # 4. DATA LOADERS
    train_transform, val_transform = get_transforms(CONFIG['img_h'], CONFIG['img_w'])
    train_ds = BiomassDataset(train_df, CONFIG['data_dir'], TARGET_COLUMNS, tab_cols, train_transform)
    val_ds = BiomassDataset(val_df, CONFIG['data_dir'], TARGET_COLUMNS, tab_cols, val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'])
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
    
    # 5. MODEL SELECTION
    # model = MultiModalConvNeXt(
    #     model_name=CONFIG['model_name'], 
    #     tabular_dim=len(tab_cols)
    # ).to(CONFIG['device'])

    model = EnvironmentalContextModel(
        model_name=CONFIG['model_name'], 
        tabular_dim=len(tab_cols)
    ).to(CONFIG['device'])

    current_lr = CONFIG['lr']
    if CONFIG['resume'] and os.path.exists("models_checkpoints/best_local_model_contextual.pth"):
        print("Resuming from checkpoint... reducing LR for fine-tuning.")
        model.load_state_dict(torch.load("models_checkpoints/best_local_model_contextual.pth"))
        current_lr = current_lr * 0.1 # Start with 10% of original LR

    optimizer = optim.AdamW(model.parameters(), lr=current_lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    
    # Using Huber Loss for robustness against outliers (L1 behavior for large errors)
    criterion = WeightedHuberLoss(TARGET_WEIGHTS, delta=1.0)
    
    # 6. TRAINING LOOP
    history = {'train_loss': [], 'val_loss': [], 'val_metric': []}
    best_r2 = -float('inf')
    
    for epoch in range(CONFIG['epochs']):
        # Train
        model.train()
        total_train_loss = 0
        for images, tabular, targets in train_loader:
            images, tabular, targets = images.to(CONFIG['device']), tabular.to(CONFIG['device']), targets.to(CONFIG['device'])
            
            optimizer.zero_grad()
            outputs = model(images, tabular)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # --- GRADIENT CLIPPING (Prevention of Loss Spikes) ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        total_val_loss = 0
        with torch.no_grad():
            for images, tabular, targets in val_loader:
                images, tabular, targets = images.to(CONFIG['device']), tabular.to(CONFIG['device']), targets.to(CONFIG['device'])
                outputs = model(images, tabular)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()
                val_preds.append(outputs.cpu().numpy())
                val_targets.append(targets.cpu().numpy())
        
        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        r2 = competition_metric(val_targets, val_preds)
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_metric'].append(r2)
        
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Train Loss: {avg_train_loss:.4f} | Val R2: {r2:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            torch.save(model.state_dict(), "models_checkpoints/best_local_model_contextual.pth")

    # 7. PLOT RESULTS
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_metric'], label='R2 Score', color='green')
    plt.title('Validation R2')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("outputs/local_training_history.png")
    plt.show()

if __name__ == "__main__":
    train_local()
