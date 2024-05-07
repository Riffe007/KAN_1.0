# File: rms_norm.py
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        mean_square = torch.mean(x ** 2, dim=-1, keepdim=True)
        normalized_x = x / torch.sqrt(mean_square + self.eps)
        return self.scale * normalized_x
