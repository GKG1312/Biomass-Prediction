
import torch
import torch.nn as nn
import timm

class VisionOnlyBiomassModel(nn.Module):
    """
    A purely visual model that does NOT rely on metadata.
    This is designed for the scenario where test-time metadata is missing.
    """
    def __init__(self, model_name='convnext_tiny.in12k_ft_in1k', pretrained=True):
        super().__init__()
        # Using a strong backbone like ConvNeXt or ViT
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=5)
        
    def forward(self, image, tabular=None):
        # We ignore the tabular input to ensure zero-reliance on metadata
        return self.backbone(image)
