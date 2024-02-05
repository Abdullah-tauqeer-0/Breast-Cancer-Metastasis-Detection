import torch
import torch.nn as nn
from .attention import SelectiveNeighborhoodAttention
from torchvision.models import resnet50

class BreastCancerModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Backbone for feature extraction
        resnet = resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Selective Neighborhood Attention Module
        self.sna = SelectiveNeighborhoodAttention(dim=2048, kernel_size=7, num_heads=8)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.permute(0, 2, 3, 1) # [B, H, W, C] for NATTEN
        x = self.sna(x)
        x = x.permute(0, 3, 1, 2) # Back to [B, C, H, W]
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
