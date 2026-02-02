
import torch
import torch.nn as nn
import timm

class SelfAugmentedHydraModel(nn.Module):
    """
    Self-Augmented Hydra Architecture as per USER request:
    1. Backbone (Image -> Embeddings)
    2. Meta Predictor (Embeddings -> NDVI, Height, Species)
       - Species is treated as a single numeric scalar.
    3. Final Hydra Predictors (Embeddings + Predicted Meta -> 5 Biomass targets)
    """
    def __init__(self, model_name='convnext_tiny.in12k_ft_in1k', pretrained=True):
        super().__init__()
        # 1. Backbone
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        embed_dim = self.backbone.num_features
        
        # 2. Meta Predictor (Input: Embeddings -> Output: [NDVI, Height, Species])
        # We use a small MLP to predict these three values
        self.meta_predictor = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3) # [NDVI, Height, Species_Index]
        )
        
        # 3. Final Hydra Predictors (Input: Embeddings + 3 Meta Predictions)
        # We concatenate the 3 predicted values to the image embeddings
        fusion_dim = embed_dim + 3
        
        self.head_clover = nn.Sequential(nn.Linear(fusion_dim, 256), nn.GELU(), nn.Linear(256, 1))
        self.head_dead   = nn.Sequential(nn.Linear(fusion_dim, 256), nn.GELU(), nn.Linear(256, 1))
        self.head_green  = nn.Sequential(nn.Linear(fusion_dim, 256), nn.GELU(), nn.Linear(256, 1))
        self.head_gdm    = nn.Sequential(nn.Linear(fusion_dim, 256), nn.GELU(), nn.Linear(256, 1))
        self.head_total  = nn.Sequential(nn.Linear(fusion_dim, 256), nn.GELU(), nn.Linear(256, 1))
        
    def forward(self, image, return_meta=False):
        # 1. Get Image Embeddings
        embeddings = self.backbone(image)
        
        # 2. Predict Meta Features (NDVI, Height, Species)
        # These are internally generated context scalars
        pred_meta = self.meta_predictor(embeddings) # Shape: [Batch, 3]
        
        # 3. Concatenate Embeddings with Predicted Meta
        combined_input = torch.cat([embeddings, pred_meta], dim=1)
        
        # 4. Hydra Predictions
        p_clover = self.head_clover(combined_input)
        p_dead   = self.head_dead(combined_input)
        p_green  = self.head_green(combined_input)
        p_gdm    = self.head_gdm(combined_input)
        p_total  = self.head_total(combined_input)
        
        biomass_preds = torch.cat([p_clover, p_dead, p_green, p_gdm, p_total], dim=1)
        
        if return_meta:
            return biomass_preds, pred_meta
        return biomass_preds
