
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from data.dataset import BiomassDataset, get_transforms
from models.baseline import MultiModalConvNeXt
from models.contextual import EnvironmentalContextModel
from utils.metrics import TARGET_COLUMNS
import joblib

def run_local_inference():
    data_dir = r"d:\personalProject\CSIRO-Image2Biomass_Prediction\csiro-biomass"
    checkpoint_dir = "models_checkpoints"
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    
    # Identify unique images for processing to save time
    unique_test_images = test_df.drop_duplicates(subset=['image_path']).copy()
    
    # 1. Load Metadata Info (Scaler and Feature Schema from training)
    meta_path = os.path.join(checkpoint_dir, "metadata_info_local.pkl")
    if not os.path.exists(meta_path):
        print("Error: metadata_info_local.pkl not found. Please train the model first.")
        return
    meta_info = joblib.load(meta_path)
    tab_cols = meta_info['tab_cols']
    scaler = meta_info['scaler']

    # 2. Metadata Handling (Create placeholders for features missing in test set)
    # Temporal
    unique_test_images['Sampling_Date'] = pd.to_datetime(unique_test_images.get('Sampling_Date', '2015-01-01'))
    unique_test_images['DayOfYear'] = unique_test_images['Sampling_Date'].dt.dayofyear
    unique_test_images['sin_day'] = np.sin(2 * np.pi * unique_test_images['DayOfYear'] / 365.0)
    unique_test_images['cos_day'] = np.cos(2 * np.pi * unique_test_images['DayOfYear'] / 365.0)
    
    # Categorical / Missing
    for col in tab_cols:
        if col not in unique_test_images.columns:
            unique_test_images[col] = 0.0 # Standard default
            
    # Scale
    unique_test_images[tab_cols] = scaler.transform(unique_test_images[tab_cols])
    
    # 3. Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = MultiModalConvNeXt(tabular_dim=len(tab_cols)).to(device)
    model = EnvironmentalContextModel(tabular_dim=len(tab_cols)).to(device)
    model_path = os.path.join(checkpoint_dir, "best_local_model.pth")
    if not os.path.exists(model_path):
        # Fallback to ensemble fold 0 if exists
        model_path = os.path.join(checkpoint_dir, "best_model_fold_0.pth")
        
    if not os.path.exists(model_path):
        print(f"Error: No model checkpoint found at {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 4. Transforms
    _, val_transform = get_transforms(img_h=224, img_w=448) # Match local training config
    
    # 5. Dataloader
    test_ds = BiomassDataset(unique_test_images, data_dir, TARGET_COLUMNS, tab_cols, val_transform, is_test=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    # 6. Prediction
    preds = []
    with torch.no_grad():
        for images, tabular in test_loader:
            images, tabular = images.to(device), tabular.to(device)
            outputs = model(images, tabular)
            preds.append(outputs.cpu().numpy())
                
    final_preds = np.vstack(preds)
    
    # 7. Map back to Sample IDs
    image_paths = unique_test_images['image_path'].values
    result_map = {path: preds for path, preds in zip(image_paths, final_preds)}
    
    submission = []
    for idx, row in test_df.iterrows():
        img_path = row['image_path']
        target_name = row['target_name']
        target_idx = TARGET_COLUMNS.index(target_name)
        
        pred_value = result_map[img_path][target_idx]
        submission.append({
            'sample_id': row['sample_id'],
            'target': max(0.0, float(pred_value)) 
        })
        
    sub_df = pd.DataFrame(submission)
    os.makedirs("outputs", exist_ok=True)
    sub_df.to_csv("outputs/submission.csv", index=False)
    print("\nLocal Inference finished. Submission generated: outputs/submission.csv")

if __name__ == "__main__":
    run_local_inference()
