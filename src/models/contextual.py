
import torch
import torch.nn as nn
import timm

class EnvironmentalContextModel(nn.Module):
    """
    Advanced Multi-Modal model that uses an 'Environment Encoder' sub-network
    to process metadata into a rich context vector before fusion.
    """
    def __init__(self, model_name='convnext_nano.in12k_ft_in1k', pretrained=True, tabular_dim=22):
        super().__init__()
        # 1. Vision Branch
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        vis_dim = self.backbone.num_features
        
        # 2. Context Branch (The Environment Encoder)
        # This branch extracts 'meaning' from the raw CSV data (one-hot states, species, NDVI, etc.)
        self.env_encoder = nn.Sequential(
            nn.Linear(tabular_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 64) # Output is a 64-dim Context Vector
        )
        
        # 3. Fusion Head
        # Combines the Image signal and the Environment signal
        self.fusion_head = nn.Sequential(
            nn.Linear(vis_dim + 64, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 5) # 5 Biomass targets
        )
        
    def forward(self, image, tabular):
        # Extract features from both branches
        vis_feats = self.backbone(image)
        ctx_feats = self.env_encoder(tabular)
        
        # Concatenate Vision + Context
        combined = torch.cat([vis_feats, ctx_feats], dim=1)
        
        # Final prediction
        return self.fusion_head(combined)
