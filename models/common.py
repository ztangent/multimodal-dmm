from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn

class GaussianMLP(nn.Module):
    """MLP from input to Gaussian output parameters."""
    def __init__(self, x_dim, y_dim, h_dim):
        super(GaussianMLP, self).__init__()
        self.x_to_h = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU())
        self.h_to_y_mean = nn.Linear(h_dim, y_dim)
        self.h_to_y_std = nn.Sequential(
            nn.Linear(h_dim, y_dim),
            nn.Softplus())

    def forward(self, x):
        h = self.x_to_h(x)
        mean, std = self.h_to_y_mean(h), self.h_to_y_std(h)
        return mean, std
