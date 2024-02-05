import torch
import torch.nn as nn
from natten import NeighborhoodAttention2D

class SelectiveNeighborhoodAttention(nn.Module):
    def __init__(self, dim, kernel_size, num_heads):
        super().__init__()
        self.na = NeighborhoodAttention2D(dim=dim, kernel_size=kernel_size, num_heads=num_heads)
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, H, W, C]
        attn_out = self.na(x)
        
        # Selective gating mechanism based on tissue context
        gate_score = self.gate(x)
        return attn_out * gate_score + x  # Residual connection with gating
