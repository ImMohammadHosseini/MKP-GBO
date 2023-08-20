
"""

"""
import torch
import torch.nn as nn
from typing import List, Optional


class MLP(nn.Module):
    def __init__(self, 
        rowNum: int, 
        colNum: int,
        hidden_dims: List = [254, 128, 128, 64],
        device = torch.device('cpu'),
        name = 'mlp'
    ):
        super(MLP, self).__init__()
        self.name = name
        self.device = device
        
        modules = []
        input_dim = rowNum * colNum
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, out_features=h_dim),
                    nn.Softplus())
            )
            input_dim = h_dim
        self.mlp = nn.Sequential(*modules).to(self.device)
        self.outer = nn.Linear(input_dim, 1, device=self.device)
        self.sigmoid =  nn.Sigmoid()

    def forward(self, x):
        x = x.flatten()
        x = self.mlp(x)
        return self.sigmoid(self.outer(x))