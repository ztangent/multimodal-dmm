"""Shared loss functions."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def kld_gauss(mean_1, std_1, mean_2, std_2, mask=None):
    kld_element =  (2 * torch.log(std_2) - 2 * torch.log(std_1) + 
        (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
        std_2.pow(2) - 1)
    if mask is not None:
        kld_element = kld_element.masked_select(mask)
    kld =  0.5 * torch.sum(kld_element)
    return kld

def nll_bernoulli(theta, x, mask=None):
    """Returns Bernoulli negative log-likelihood (summed across inputs).

        theta : torch.tensor of shape (T, B, D, ...)
            Probability of heads
        x : torch.tensor of shape (T, B, D, ...)
            Tensor of observations
        mask : torch.tensor of shape (T, B)

        Here, T = n_timesteps, B = batch_size, and (D, ...) are the input dims.
    """
    if mask is None:
        mask = 1 - torch.isnan(x)
    else:
        shape = list(mask.shape) + [1] * (x.dim() - mask.dim())
        mask = (1 - torch.isnan(x)) * mask.view(*shape)
    theta = theta.masked_select(mask)
    x = x.masked_select(mask)
    nll = F.binary_cross_entropy(theta, x, reduction='sum')
    return nll

def nll_categorical(probs, x, mask=None):
    """Returns categorical negative log-likelihood (summed across inputs).

        probs : torch.tensor of shape (T, B, K, D, ...)
            Probability of heads
        x : torch.tensor of shape (T, B, D, ...)
            Tensor of observed category labels (not one-hot).
        mask : torch.tensor of shape (T, B)

        Here, T = n_timesteps, B = batch_size, K = n_categories,
        and (D, ...) are the input dims.
    """
    if mask is None:
        mask = 1 - torch.isnan(x)
    else:
        shape = list(mask.shape) + [1] * (x.dim() - mask.dim())
        mask = (1 - torch.isnan(x)) * mask.view(*shape)
    # Mask probs and reshape into correct format for F.nll_loss
    probs = torch.stack([probs[:,:,k].masked_select(mask)
                         for k in probs.shape[2]], dim=-1)
    x = x.masked_select(mask)
    nll = F.nll_loss(probs, x, reduction='sum')
    return nll

def nll_gauss(mean, std, x, mask=None):
    """Returns Gaussian negative log-likelihood (summed across inputs).

        mean : torch.tensor of shape (T, B, D, ...)
        std : torch.tensor of shape (T, B, D, ...)
        x : torch.tensor of shape (T, B, D, ...)
        mask : torch.tensor of shape (T, B)

        Here, T = n_timesteps, B = batch_size, and (D, ...) are the input dims.
    """
    if mask is None:
        mask = 1 - torch.isnan(x)
    else:
        shape = list(mask.shape) + [1] * (x.dim() - mask.dim())
        mask = (1 - torch.isnan(x)) * mask.view(*shape)
    x = torch.tensor(x)
    x[torch.isnan(x)] = 0.0
    nll_element = ( ((x-mean).pow(2)) / (2 * std.pow(2)) + std.log() +
                    math.log(math.sqrt(2 * math.pi)) )
    nll_element = nll_element.masked_select(mask)
    nll = torch.sum(nll_element)
    return nll
