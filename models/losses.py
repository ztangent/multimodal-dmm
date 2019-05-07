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
    if mask is None:
        mask = 1 - torch.isnan(x)
    else:
        shape = list(mask.shape) + [1] * (x.dim() - mask.dim())
        mask = (1 - torch.isnan(x)) * mask.view(*shape)
    theta = theta.masked_select(mask)
    x = x.masked_select(mask)
    nll = F.binary_cross_entropy(theta, x, reduction='sum')
    return nll

def nll_gauss(mean, std, x, mask=None):
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
