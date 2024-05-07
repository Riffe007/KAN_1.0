# File: multihead_kan_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficient_kan import KANLinear
from utils.rope import RoPE

class MultiheadKANAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, rotation_matrix):
        super(MultiheadKANAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

        self.qkv_linear = KANLinear(hidden_size, hidden_size * 3)
        self.out_linear = nn.Linear(hidden_size, hidden_size)
        self.position_emb = RoPE(rotation_matrix)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_linear(x)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.transpose(1, 2)
        queries, keys, values = torch.chunk(qkv, 3, dim=-1)

        queries, keys = self.position_emb(queries, keys)

        scores = torch.matmul(queries, keys.transpose(-2, -1))
        scores = scores / (self.head_dim ** 0.5)
        attention = F.softmax(scores, dim=-1)
        context = torch.matmul(attention, values)

        context = context.transpose(1, 2).reshape(batch_size, seq_length, -1)
        return self.out_linear(context)
