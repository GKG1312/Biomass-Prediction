
import torch
import torch.nn as nn
import timm

class AdvancedHydraModel(nn.Module):
    """
    Advanced Self-Augmented Hydra Architecture:
    1. Hierarchical Backbone (Swin/ConvNeXt)
    2. Split Meta-Predictor:
       - Regression: [NDVI, Height]
       - Classification: Species (Categorical)
    3. Feature Fusion: Visual + Metadata Embeddings
    4. 5 Hydra Heads for Biomass
    """
    def __init__(self, model_name='swin_tiny_patch4_window7_224', num_species=15, pretrained=True):
        super().__init__()
        # 1. Backbone
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        embed_dim = self.backbone.num_features
        
        # 2. Advanced Meta Predictors
        self.meta_regressor = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 2) # [NDVI, Height]
        )
        
        self.species_classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_species) # Categorical Species
        )
        
        # 3. Species Embedding for Fusion
        # We turn the predicted species back into a high-dim vector to help the hydra heads
        self.species_embedding = nn.Embedding(num_species, 32)
        
        # 4. Final Hydra Predictors
        # Input: Visual (embed_dim) + Reg Meta (2) + Species Embedding (32)
        fusion_dim = embed_dim + 2 + 32
        
        self.head_clover = nn.Sequential(nn.Linear(fusion_dim, 256), nn.GELU(), nn.Linear(256, 1))
        self.head_dead   = nn.Sequential(nn.Linear(fusion_dim, 256), nn.GELU(), nn.Linear(256, 1))
        self.head_green  = nn.Sequential(nn.Linear(fusion_dim, 256), nn.GELU(), nn.Linear(256, 1))
        self.head_gdm    = nn.Sequential(nn.Linear(fusion_dim, 256), nn.GELU(), nn.Linear(256, 1))
        self.head_total  = nn.Sequential(nn.Linear(fusion_dim, 256), nn.GELU(), nn.Linear(256, 1))
        
    def forward(self, image, return_meta=False):
        # 1. Visual Features
        vis_features = self.backbone(image) # [Batch, embed_dim]
        
        # 2. Predict Meta
        pred_reg = self.meta_regressor(vis_features)  # [NDVI, Height]
        pred_species_logits = self.species_classifier(vis_features) # [num_species]
        
        # 3. Prepare contextual features for biomass
        # At training, we could use GT species, but for "Self-Augmented" 
        # we use the most likely predicted species.
        best_species = torch.argmax(pred_species_logits, dim=1)
        spec_feat = self.species_embedding(best_species)
        
        combined_input = torch.cat([vis_features, pred_reg, spec_feat], dim=1)
        
        # 4. Hydra Predictions
        out = torch.cat([
            self.head_clover(combined_input),
            self.head_dead(combined_input),
            self.head_green(combined_input),
            self.head_gdm(combined_input),
            self.head_total(combined_input)
        ], dim=1)
        
        if return_meta:
            return out, pred_reg, pred_species_logits
        return out
