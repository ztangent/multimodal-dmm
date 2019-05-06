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
    def __init__(self, z_dim, h_dim, min_std=0):
        super(GaussianGTF, self).__init__()
        self.min_std = min_std
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
        z_std = self.z_to_std(z_nonlin) + self.min_std
        z_mean = (1-gate) * z_lin + gate * z_nonlin
        return z_mean, z_std

class Conv(nn.Module):
    """Convolutional layer with optional batch norm and ReLU."""
    def __init__(self, n_channels, n_kernels,
                 kernel_size=3, stride=2, padding=1, last=False):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(
            n_channels, n_kernels,
            kernel_size, stride, padding
        )
        if not last:
            self.net = nn.Sequential(
                self.conv,
                nn.BatchNorm2d(n_kernels),
                nn.ReLU()
            )
        else:
            self.net = self.conv
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        return self.net(x)

class Deconv(nn.Module):
    """De-convolutional layer with optional batch norm and ReLU."""
    def __init__(self, n_channels, n_kernels,
                 kernel_size=4, stride=2, padding=1, last=False):
        super(Deconv, self).__init__()
        self.deconv = nn.ConvTranspose2d(
            n_channels, n_kernels,
            kernel_size, stride, padding
        )
        if not last:
            self.net = nn.Sequential(
                self.deconv,
                nn.BatchNorm2d(n_kernels),
                nn.ReLU()
            )
        else:
            self.net = self.deconv
        nn.init.xavier_uniform_(self.deconv.weight)

    def forward(self, x):
        return self.net(x)
    
class ImageEncoder(nn.Module):
    """Convolutional encoder for images."""
    def __init__(self, z_dim,
                 img_size=64, n_channels=3, n_kernels=64, n_layers=3):
        super(ImageEncoder, self).__init__()
        self.feat_size = img_size // 2**n_layers
        self.feat_dim = self.feat_size ** 2 * n_kernels

        self.conv_stack = nn.Sequential(
            *([Conv(n_channels, n_kernels // 2**(n_layers-1))] +
              [Conv(n_kernels//2**(n_layers-l), n_kernels//2**(n_layers-l-1))
               for l in range(1, n_layers-1)] +
              [Conv(n_kernels // 2, n_kernels, last=True)])
        )

        self.feat_to_z_mean = nn.Linear(self.feat_dim, z_dim)
        self.feat_to_z_std = nn.Sequential(
            nn.Linear(self.feat_dim, z_dim),
            nn.Softplus()
        )

        nn.init.xavier_uniform_(self.feat_to_z_mean.weight)
        nn.init.xavier_uniform_(self.feat_to_z_std[0].weight)
        
    def forward(self, x):
        feats = self.conv_stack(x)
        z_mean = self.feat_to_z_mean(feats.view(-1, self.feat_dim))
        z_std = self.feat_to_z_std(feats.view(-1, self.feat_dim))
        return z_mean, z_std

class ImageDecoder(nn.Module):
    """De-convolutional decoder for images."""
    def __init__(self, z_dim,
                 img_size=64, n_channels=3, n_kernels=64, n_layers=3):
        super(ImageDecoder, self).__init__()
        self.feat_size = img_size // 2**n_layers
        self.feat_dim = self.feat_size ** 2 * n_kernels
        self.feat_shape = (n_kernels, self.feat_size, self.feat_size)

        self.z_to_feat = nn.Sequential(
            nn.Linear(z_dim, self.feat_dim),
            nn.ReLU()
        )

        self.deconv_stack = nn.Sequential(
            *([Deconv(n_kernels // 2**l, n_kernels // 2**(l+1))
               for l in range(n_layers-1)] +
              [Deconv(n_kernels // 2**(n_layers-1), n_channels, last=True)] +
              [nn.Sigmoid()]
            )
        )

        nn.init.xavier_uniform_(self.z_to_feat[0].weight)

    def forward(self, z):
        feats = self.z_to_feat(z).view(-1, *self.feat_shape)
        x_mean = self.deconv_stack(feats)
        x_std = (x_mean * (1-x_mean)).pow(0.5)
        return x_mean, x_std

class ImageEncoderFC(nn.Module):
    """Fully-connected encoder for images."""
    def __init__(self, z_dim, h_dim=256, img_size=64, n_channels=3):
        super(ImageEncoderFC, self).__init__()
        self.img_dim = img_size * img_size * n_channels
        self.feat_dim = h_dim

        self.fc_stack = nn.Sequential(
            nn.Linear(self.img_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )

        self.feat_to_z_mean = nn.Linear(self.feat_dim, z_dim)
        self.feat_to_z_std = nn.Sequential(
            nn.Linear(self.feat_dim, z_dim),
            nn.Softplus()
        )

        nn.init.xavier_uniform_(self.feat_to_z_mean.weight)
        nn.init.xavier_uniform_(self.feat_to_z_std[0].weight)
        for layer in self.fc_stack:
            if type(layer) == nn.Linear:
                nn.init.xavier_uniform_(layer.weight)
        
    def forward(self, x):
        feats = self.fc_stack(x.view(-1, self.img_dim))
        z_mean = self.feat_to_z_mean(feats.view(-1, self.feat_dim))
        z_std = self.feat_to_z_std(feats.view(-1, self.feat_dim))
        return z_mean, z_std

class ImageDecoderFC(nn.Module):
    """Fully-connected decoder for images."""
    def __init__(self, z_dim, h_dim=256, img_size=64, n_channels=3):
        super(ImageDecoderFC, self).__init__()
        self.img_shape = (n_channels, img_size, img_size)
        self.img_dim = img_size * img_size * n_channels
        self.feat_dim = h_dim

        self.fc_stack = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, self.img_dim),
            nn.Sigmoid()
        )
        
        self.z_to_feat = nn.Sequential(
            nn.Linear(z_dim, self.feat_dim),
            nn.ReLU()
        )

        nn.init.xavier_uniform_(self.z_to_feat[0].weight)
        for layer in self.fc_stack:
            if type(layer) == nn.Linear:
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, z):
        feats = self.z_to_feat(z)
        x_mean = self.fc_stack(feats).reshape(-1, *self.img_shape)
        x_std = (x_mean * (1-x_mean)).pow(0.5)
        return x_mean, x_std
