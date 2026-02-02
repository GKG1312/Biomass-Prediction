
import torch
import torch.nn as nn
import timm

class SelfAugmentedDINOv2(nn.Module):
    """
    Self-Augmented DINOv2 model.
    It predicts its own metadata (NDVI, Height) from images and uses 
    those internal predictions to guide biomass estimation.
    This eliminates the need for test-time metadata.
    """
    def __init__(self, model_name='vit_base_patch14_dinov2.lvd142m', pretrained=True, num_meta=2):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, dynamic_img_size=True)
        embed_dim = self.backbone.num_features
        
        # Internal Metadata Reconstruction (e.g., NDVI and Height)
        self.meta_predictor = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, num_meta)
        )
        
        # Context Encoder (Processes the predicted metadata)
        self.env_encoder = nn.Sequential(
            nn.Linear(num_meta, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64)
        )
        
        # Fusion Head
        self.head = nn.Sequential(
            nn.Linear(embed_dim + 64, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 5)
        )
        
    def forward(self, image, return_meta=False):
        vis_feats = self.backbone(image)
        
        # Generate internal metadata predictions
        pred_meta = self.meta_predictor(vis_feats)
        
        # Map internal meta to context space
        ctx_feats = self.env_encoder(pred_meta)
        
        # Combine visual features with predicted contextual hints
        combined = torch.cat([vis_feats, ctx_feats], dim=1)
        biomass = self.head(combined)
        
        if return_meta:
            return biomass, pred_meta
        return biomass
