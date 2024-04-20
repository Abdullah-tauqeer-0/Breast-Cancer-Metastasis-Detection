import torch
import torch.nn as nn
from natten import NeighborhoodAttention2D

class SelectiveNeighborhoodAttention(nn.Module):
    """
    Selective Neighborhood Attention Module.
    Combines Neighborhood Attention with a tissue-aware gating mechanism.
    """
    def __init__(self, dim, kernel_size, num_heads, qkv_bias=True, dilation=None):
        super().__init__()
        self.na = NeighborhoodAttention2D(
            dim=dim, 
            kernel_size=kernel_size, 
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            dilation=dilation
        )
        
        # Gating mechanism to weigh attention based on local context
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Input x: [B, H, W, C]
        
        # 1. Compute Neighborhood Attention
        attn_out = self.na(x)
        
        # 2. Compute Gate Score
        gate_score = self.gate(x)
        
        # 3. Selective Aggregation (Residual + Gated Attention)
        out = x + self.dropout(attn_out * gate_score)
        
        return self.norm(out)
