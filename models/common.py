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

class GaussianGTF(nn.Module):
    """GRU-like latent space gated transition function (GTF)."""
    def __init__(self, z_dim, h_dim):
        super(GaussianGTF, self).__init__()
        self.z_to_gate = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim),
            nn.Sigmoid())
        self.z_lin = nn.Linear(z_dim, z_dim)
        self.z_nonlin = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim))
        self.z_to_std = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.Softplus())        

    def forward(self, z):
        gate = self.z_to_gate(z)
        z_lin = self.z_lin(z)
        z_nonlin = self.z_nonlin(z)
        z_std = self.z_to_std(z_nonlin)
        z_mean = (1-gate) * z_lin + gate * z_nonlin
        return z_mean, z_std
