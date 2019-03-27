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
        self.h_to_y_std = nn.Sequential(nn.Linear(h_dim, y_dim), nn.Softplus())

    def forward(self, x):
        h = self.x_to_h(x)
        mean, std = self.h_to_y_mean(h), self.h_to_y_std(h)
        return mean, std

class GaussianGRU(nn.Module):
    """GRU-like latent space transition function."""
    def __init__(self, z_dim):
        super(GaussianGRU, self).__init__()
        self.split = z_dim // 2
        x_dim, h_dim = self.split, z_dim - self.split
        self.gru = torch.nn.GRUCell(x_dim, h_dim)
        self.h_to_x_mean = nn.Linear(h_dim, x_dim)
        self.h_to_z_std = nn.Sequential(nn.Linear(h_dim, z_dim), nn.Softplus())

    def forward(self, z):
        # Split latent state into input and hidden state
        x, h = z[:,:self.split], z[:,self.split:]
        # Compute next hidden state
        h_next = self.gru(x, h)
        # Compute next input state
        x_next = self.h_to_x_mean(h_next)
        # Concatenate into mean of next latent state
        z_mean = torch.cat([x_next, h_next], dim=-1)
        # Compute standard deviation of next latent state
        z_std = self.h_to_z_std(h_next)
        return z_mean, z_std
