
import torch
import torch.nn as nn
import timm

class MultiModalConvNeXt(nn.Module):
    def __init__(self, model_name='convnext_base', pretrained=True, tabular_dim=3):
        super().__init__()
        # Image Branch (Pre-trained Backbone)
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        backbone_out_dim = self.backbone.num_features
        
        # Metadata Branch (Directly added to embeddings)
        # We concatenate the numeric tabular features directly
        
        # Fusion Head (Combines Visual + N Tabular features)
        self.fusion_head = nn.Sequential(
            nn.Linear(backbone_out_dim + tabular_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 5) # 5 Biomass targets
        )
        
    def forward(self, image, tabular):
        img_features = self.backbone(image)
        # Tabular is now concatenated directly
        combined = torch.cat([img_features, tabular], dim=1)
        output = self.fusion_head(combined)
        return output
