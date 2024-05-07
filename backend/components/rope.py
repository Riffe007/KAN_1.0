# File: rope.py
import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self, rotation_matrix):
        super(RoPE, self).__init__()
        self.rotation_matrix = rotation_matrix

    def forward(self, queries, keys):
        # Apply the rotational matrix to both queries and keys
        queries_rotated = torch.einsum('bhsd,md->bhsm', queries, self.rotation_matrix)
        keys_rotated = torch.einsum('bhsd,md->bhsm', keys, self.rotation_matrix)
        return queries_rotated, keys_rotated
