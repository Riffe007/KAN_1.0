# In file: models/kan_block.py
import torch
import torch.nn as nn
from components.multihead_kan_attention import MultiheadKANAttention
from moe_kan_layer import MoeKANLayer
from components.rms_norm import RMSNorm

class KANBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, window_size, d_ff, num_experts, n_experts_per_token, rotation_matrix):
        super(KANBlock, self).__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.attention = MultiheadKANAttention(hidden_size, num_heads, window_size, num_experts, n_experts_per_token, rotation_matrix)
        self.moe = MoeKANLayer(hidden_size, d_ff, num_experts, n_experts_per_token)

    def forward(self, x):
        x1 = self.attention(self.norm1(x))
        x += x1
        x2 = self.moe(self.norm2(x))
        return x + x2
