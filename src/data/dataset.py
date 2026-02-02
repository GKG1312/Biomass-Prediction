
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class BiomassDataset(Dataset):
    def __init__(self, df, img_dir, target_columns, tabular_columns, transform=None, is_test=False):
        self.df = df
        self.img_dir = img_dir
        self.target_columns = target_columns
        self.tabular_columns = tabular_columns
        self.transform = transform
        self.is_test = is_test
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load Image
        img_path = os.path.join(self.img_dir, row['image_path'])
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        if self.transform:
            image = self.transform(image=image)['image']
            
        # Tabular Data: Using pre-processed numeric columns
        # tabular = torch.tensor(row[self.tabular_columns].values.astype(np.float32), dtype=torch.float32)

        # Note: DayOfYear should be converted to cyclical sin/cos for better context
        tabular_values = row[self.tabular_columns].values.astype(np.float32)
        tabular = torch.tensor(tabular_values, dtype=torch.float32)
        
        if self.is_test:
            return image, tabular
        
        # Ground Truth Targets
        targets = torch.tensor(row[self.target_columns].values.astype(np.float32), dtype=torch.float32)
        return image, tabular, targets

def get_transforms(img_h=392, img_w=784):
    train_transform = A.Compose([
        A.Resize(img_h, img_w),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(img_h, img_w),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    return train_transform, val_transform
