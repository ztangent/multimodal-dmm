"""Abstract base class for deep generative time series (DGTS) models."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import math
import torch
import torch.nn as nn

class MultiDGTS(nn.Module):
    """Abstract base class for deep generative time series (DGTS) models."""
    
    def product_of_experts(self, mean, std, mask=None, eps=1e-8):
        """
        Return parameters for product of independent Gaussian experts.
        See https://arxiv.org/pdf/1410.7827.pdf for equations.

        mean : torch.tensor
            (M, B, D) for M experts, batch size B, and D latent dims OR
            (M, T, B, D) with an optional time dimension
        std : torch.tensor
            (M, B, D) for M experts, batch size B, and D latent dims OR
            (M, T, B, D) with an optional time dimension
        mask : torch.tensor
            (M, B) for M experts and batch size B
            (M, T, B) with an optional time dimension
        """
        # Square std and add numerical constant for stability
        var = std.pow(2) + eps
        # Precision matrix of i-th Gaussian expert (T = 1/sigma^2)
        T = 1. / var
        # Set missing data to zero so they are excluded from calculation
        if mask is None:
            mask = 1 - torch.isnan(var).any(dim=-1)
        T = T * mask.float().unsqueeze(-1)
        mean = mean * mask.float().unsqueeze(-1)
        product_mean = torch.sum(mean * T, dim=0) / torch.sum(T, dim=0)
        product_mean[torch.isnan(product_mean)] = 0.0
        product_std = (1. / torch.sum(T, dim=0)).pow(0.5)
        return product_mean, product_std

    def mean_of_experts(self, mean, std, mask=None):
        """
        Return mean and standard deviation of a mixture of Gaussian experts

        mean : torch.tensor
            (M, B, D) for M experts, batch size B, and D latent dims
        var : torch.tensor
            (M, B, D) for M experts, batch size B, and D latent dims
        mask : torch.tensor
            (M, B) for M experts and batch size B
        """
        # Set missing data to zero so they are excluded from calculation
        if mask is None:
            mask = 1 - torch.isnan(std).any(dim=-1)
        mean = mean * mask.float().unsqueeze(-1)
        var = std.pow(2) * mask.float().unsqueeze(-1)
        sum_mean = torch.mean(mean, dim=0)
        sum_var = torch.mean(var, dim=0) + (torch.mean(mean.pow(2), dim=0) -
                                            sum_mean.pow(2))
        sum_std = sum_var.pow(0.5)
        return sum_mean, sum_std
    
    def loss(self, inputs, infer, prior, outputs, mask=1,
             kld_mult=1.0, rec_mults={}, avg=False):
        loss = 0.0
        loss += kld_mult * self.kld_loss(infer, prior, mask)
        loss += self.rec_loss(inputs, outputs, mask, rec_mults)
        if avg:
            if type(mask) is torch.Tensor:
                n_data = torch.sum(mask)
            else:
                n_data = inputs[self.modalities[-1]].numel()
            loss /= n_data
        return loss
    
    def kld_loss(self, infer, prior, mask=None):
        """KLD loss between inferred and prior z."""
        infer_mean, infer_std = infer
        prior_mean, prior_std = prior
        return self._kld_gauss(infer_mean, infer_std,
                               prior_mean, prior_std, mask)

    def rec_loss(self, inputs, outputs, mask=None, rec_mults={}):
        """Input reconstruction loss."""
        loss = 0.0
        out_mean, out_std = outputs
        for m in self.modalities:
            if m not in inputs:
                continue
            mult = 1.0 if m not in rec_mults else rec_mults[m]
            loss += mult * self._nll_gauss(out_mean[m], out_std[m],
                                           inputs[m], mask)
        return loss
            
    def _sample_gauss(self, mean, std):
        """Use std to sample."""
        eps = torch.FloatTensor(std.size()).to(self.device).normal_()
        return eps.mul(std).add_(mean)

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2, mask=None):
        """Use std to compute KLD"""
        kld_element =  (2 * torch.log(std_2) - 2 * torch.log(std_1) + 
            (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
            std_2.pow(2) - 1)
        if mask is not None:
            kld_element = kld_element.masked_select(mask)
        kld =  0.5 * torch.sum(kld_element)
        return kld

    def _nll_bernoulli(self, theta, x, mask=None):
        nll_element = x*torch.log(theta) + (1-x)*torch.log(1-theta)
        if mask is None:
            mask = 1 - torch.isnan(x)
        else:
            mask = mask * (1 - torch.isnan(x))
        nll_element = nll_element.masked_select(mask)
        return torch.sum(nll_element)

    def _nll_gauss(self, mean, std, x, mask=None):
        if mask is None:
            mask = 1 - torch.isnan(x)
        else:
            mask = mask * (1 - torch.isnan(x))
        x = torch.tensor(x)
        x[torch.isnan(x)] = 0.0
        nll_element = ( ((x-mean).pow(2)) / (2 * std.pow(2)) + std.log() +
                        math.log(math.sqrt(2 * math.pi)) )
        nll_element = nll_element.masked_select(mask)
        nll = torch.sum(nll_element)
        return(nll)
