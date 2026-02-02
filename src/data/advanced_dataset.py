
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class AdvancedBiomassDataset(Dataset):
    """
    Dataset for Advanced Hydra:
    Returns:
    - Image
    - Biomass Targets (5)
    - Reg Meta: [NDVI, Height]
    - Species Index (for CrossEntropyLoss)
    """
    def __init__(self, df, img_dir, target_cols, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.target_cols = target_cols
        self.transform = transform
        
        # Consistent mapping for all folds
        all_unique_species = sorted(df['Species'].unique()) if 'Species' in df.columns else []
        self.species_map = {s: i for i, s in enumerate(all_unique_species)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Load Image
        img_path = os.path.join(self.img_dir, row['image_path'])
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        
        if self.transform:
            image = self.transform(image=image)['image']
            
        # 2. Targets
        biomass = torch.tensor(row[self.target_cols].values.astype(np.float32))
        
        # 3. Meta Regression [NDVI, Height]
        reg_meta = torch.tensor([
            float(row['Pre_GSHH_NDVI']), 
            float(row['Height_Ave_cm'])
        ], dtype=torch.float32)
        
        # 4. Meta Classification [Species Index]
        species_idx = torch.tensor(self.species_map.get(row['Species'], 0), dtype=torch.long)
        
        return image, biomass, reg_meta, species_idx
