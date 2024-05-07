# File: kan_former.py
import sys
import os
import torch
import torch.nn as nn
from feed_forward import FeedForward
from kan_block import KANBlock

class KANFormer(nn.Module):
    def __init__(self, config):
        super(KANFormer, self).__init__()
        self.embedding = nn.Embedding(config['vocabulary_size'], config['hidden_size'])
        self.blocks = nn.ModuleList([
            KANBlock(config) for _ in range(config['num_layers'])
        ])
        self.out = nn.Linear(config['hidden_size'], config['vocabulary_size'])

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        return self.out(x)

if __name__ == "__main__":
    # Example configuration
    config = {
        'vocabulary_size': 10000,
        'hidden_size': 512,
        'num_layers': 6,
        'num_heads': 8,
        'dropout': 0.1,
        'max_length': 512,
        'num_experts': 10,
        'n_experts_per_token': 2,
        'd_ff': 2048
    }
    model = KANFormer(config)
    print(model)
