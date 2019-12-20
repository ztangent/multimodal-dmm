from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from builtins import range
import torch
import torch.nn as nn

class CategoricalMLP(nn.Module):
    """MLP from input to categorical output."""
    def __init__(self, in_dim, out_dim, h_dim):
        super(CategoricalMLP, self).__init__()
        self.in_to_h = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU())
        self.h_to_out = nn.Sequential(
            nn.Linear(h_dim, out_dim),
            nn.Softmax(dim=1))

    def forward(self, x):
        h = self.in_to_h(x)
        probs = self.h_to_out(h)
        return (probs,)

class GaussianMLP(nn.Module):
    """MLP from input to Gaussian output parameters."""
    def __init__(self, in_dim, out_dim, h_dim, min_std=1e-3):
        super(GaussianMLP, self).__init__()
        self.min_std = min_std
        self.in_to_h = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU())
        self.h_to_mean = nn.Linear(h_dim, out_dim)
        self.h_to_std = nn.Sequential(
            nn.Linear(h_dim, out_dim),
            nn.Softplus())

    def forward(self, x):
        h = self.in_to_h(x)
        mean, std = self.h_to_mean(h), (self.h_to_std(h) + self.min_std)
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
    def __init__(self, z_dim, gauss_out=True,
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

        self.gauss_out = gauss_out
        if gauss_out:
            self.feat_to_z_mean = nn.Linear(self.feat_dim, z_dim)
            self.feat_to_z_std = nn.Sequential(
                nn.Linear(self.feat_dim, z_dim),
                nn.Softplus()
            )

            nn.init.xavier_uniform_(self.feat_to_z_mean.weight)
            nn.init.xavier_uniform_(self.feat_to_z_std[0].weight)

    def forward(self, x):
        feats = self.conv_stack(x)
        if not self.gauss_out:
            return feats
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
        probs = self.deconv_stack(feats)
        return (probs,)

class AudioConv(nn.Module):
    """1D convolutional layer with optional batch norm and ReLU."""
    def __init__(self, n_channels, n_kernels,
                 kernel_size=3, stride=2, padding=1, last=False):
        super(AudioConv, self).__init__()
        self.conv = nn.Conv1d(
            n_channels, n_kernels,
            kernel_size, stride, padding
        )
        if not last:
            self.net = nn.Sequential(
                self.conv,
                nn.BatchNorm1d(n_kernels),
                nn.ReLU()
            )
        else:
            self.net = self.conv
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        return self.net(x)

class AudioDeconv(nn.Module):
    """1D de-convolutional layer with optional batch norm and ReLU."""
    def __init__(self, n_channels, n_kernels,
                 kernel_size=3, stride=2, padding=1, last=False):
        super(AudioDeconv, self).__init__()
        self.deconv = nn.ConvTranspose1d(
            n_channels, n_kernels,
            kernel_size, stride, padding
        )
        if not last:
            self.net = nn.Sequential(
                self.deconv,
                nn.BatchNorm1d(n_kernels),
                nn.ReLU()
            )
        else:
            self.net = self.deconv
        nn.init.xavier_uniform_(self.deconv.weight)

    def forward(self, x):
        return self.net(x)

class AudioEncoder(nn.Module):
    """GRU encoder for segments of raw audio samples."""
    def __init__(self, z_dim, gauss_out=True, h_dim=256, subseg_len=256):
        super(AudioEncoder, self).__init__()
        self.subseg_len = subseg_len
        self.feat_dim = h_dim
        self.gru = nn.GRU(subseg_len, h_dim, batch_first=True)
        self.gauss_out = gauss_out
        if gauss_out:
            self.feat_to_z_mean = nn.Linear(h_dim, z_dim)
            self.feat_to_z_std = nn.Sequential(
                nn.Linear(h_dim, z_dim),
                nn.Softplus()
            )

    def forward(self, x):
        # Reshape to (batch_dim, ..., subseg_len)
        x = x.view(x.shape[0], -1, self.subseg_len)
        _, h_out = self.gru(x)
        feats = h_out[-1]
        if not self.gauss_out:
            return feats
        z_mean = self.feat_to_z_mean(feats.view(-1, self.feat_dim))
        z_std = self.feat_to_z_std(feats.view(-1, self.feat_dim))
        return z_mean, z_std

class AudioDecoder(nn.Module):
    """GRU decoder for segments of raw audio samples."""
    def __init__(self, z_dim, h_dim=256, subseg_len=256, n_subsegs=5):
        super(AudioDecoder, self).__init__()
        self.subseg_len = subseg_len
        self.n_subsegs = n_subsegs
        self.z_to_h0 = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU()
        )
        self.z_to_x0 = nn.Sequential(
            nn.Linear(z_dim, subseg_len),
            nn.ReLU()
        )
        self.gru = nn.GRU(subseg_len, h_dim, batch_first=True)
        self.h_to_mean = nn.Linear(h_dim, subseg_len)
        self.h_to_std = nn.Sequential(
            nn.Linear(h_dim, subseg_len),
            nn.Softplus())

    def forward(self, z):
        # Reshape to (1, batch_dim, h_dim)
        h = self.z_to_h0(z).unsqueeze(0)
        # Reshape to (batch_dim, 1, subseg_len)
        x_in = self.z_to_x0(z).unsqueeze(1)
        x_outs, x_stds = []
        # Auto-regressively generate the sequence of subsegments
        for seg in range(self.n_subsegs):
            _, h = self.gru(x_in, h)
            x_out = self.h_to_mean(h[-1])
            x_std = self.h_to_std(h[-1])
            x_outs.append(x_out)
            x_stds.append(x_in)
            x_in = x_out
        out_mean = torch.stack(x_outs, dim=-1)
        out_std = torch.stack(x_stds, dim=-1)
        return (out_mean, out_std)
