
import torch
import torch.nn as nn
import timm

class HydraVisionModel(nn.Module):
    """
    A multi-head vision model. 
    One backbone (Feature Extractor) and 5 separate heads (Predictors).
    This allows each head to specialize in one biomass target without 
    interference from the gradients of other targets.
    """
    def __init__(self, model_name='convnext_tiny.in12k_ft_in1k', pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        embed_dim = self.backbone.num_features
        
        # 5 Separate Heads
        # Each head is a mini-MLP dedicated to one target
        self.head_clover = nn.Sequential(nn.Linear(embed_dim, 256), nn.GELU(), nn.Linear(256, 1))
        self.head_dead   = nn.Sequential(nn.Linear(embed_dim, 256), nn.GELU(), nn.Linear(256, 1))
        self.head_green  = nn.Sequential(nn.Linear(embed_dim, 256), nn.GELU(), nn.Linear(256, 1))
        self.head_gdm    = nn.Sequential(nn.Linear(embed_dim, 256), nn.GELU(), nn.Linear(256, 1))
        self.head_total  = nn.Sequential(nn.Linear(embed_dim, 256), nn.GELU(), nn.Linear(256, 1))
        
    def forward(self, image, tabular=None):
        features = self.backbone(image)
        
        # Collect predictions from each head
        p_clover = self.head_clover(features)
        p_dead   = self.head_dead(features)
        p_green  = self.head_green(features)
        p_gdm    = self.head_gdm(features)
        p_total  = self.head_total(features)
        
        # Concatenate back to [Batch, 5]
        return torch.cat([p_clover, p_dead, p_green, p_gdm, p_total], dim=1)
