
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class SelfAugmentedBiomassDataset(Dataset):
    """
    Dataset for Self-Augmented training.
    Returns:
    - Image
    - Biomass Targets (5)
    - Metadata: [NDVI, Height, Species_Index]
    """
    def __init__(self, df, img_dir, target_columns, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.target_columns = target_columns
        self.transform = transform
        
        # Consistent mapping for species indices
        self.species_list = sorted(df['Species'].unique())
        self.species_map = {s: i for i, s in enumerate(self.species_list)}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load Image
        img_path = os.path.join(self.img_dir, row['image_path'])
        image = np.array(Image.open(img_path).convert('RGB'))
        
        if self.transform:
            image = self.transform(image=image)['image']
            
        # Biomass Targets
        biomass = torch.tensor(row[self.target_columns].values.astype(np.float32), dtype=torch.float32)
        
        # Meta Ground Truth: [NDVI, Height, Species_Index]
        # Species is cast to float32 to match NDVI/Height in a single tensor
        species_idx = float(self.species_map[row['Species']])
        meta = torch.tensor([row['Pre_GSHH_NDVI'], row['Height_Ave_cm'], species_idx], dtype=torch.float32)
        
        return image, biomass, meta
