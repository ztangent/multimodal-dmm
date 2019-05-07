"""Abstract base class for deep generative time series (DGTS) models."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn

from . import losses

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

    def step(self, inputs, mask, kld_mult, rec_mults, targets=None, **kwargs):
        """Custom training step for multimodal training paradigm."""
        # Get rid of unrecognized modalities
        inputs = {m : inputs[m] for m in inputs if m in self.modalities}
        # If targets not provided, assume inputs are targets
        if targets == None:
            targets = inputs
        loss = 0
        # Compute negative ELBO loss for individual modalities
        for m in self.modalities:
            infer, prior, recon = self.forward({m : inputs[m]}, **kwargs)
            loss += self.loss({m : targets[m]}, infer, prior, recon, mask,
                              kld_mult, rec_mults)
        # Compute negative ELBO loss for all modalities
        if len(self.modalities) > 1:
            infer, prior, recon = self.forward(inputs, **kwargs)
            loss += self.loss(targets, infer, prior, recon, mask,
                              kld_mult, rec_mults)
        return loss
        
    def loss(self, inputs, infer, prior, recon, mask=1,
             kld_mult=1.0, rec_mults={}, avg=False):
        """Computes weighted sum of KLD loss and reconstruction loss."""
        loss = 0.0
        loss += kld_mult * self.kld_loss(infer, prior, mask)
        loss += self.rec_loss(inputs, recon, mask, rec_mults)
        if avg:
            if type(mask) is torch.Tensor:
                n_data = torch.sum(mask)
            else:
                shape = inputs[self.modalities[-1]].shape
                n_data = (shape[0] * shape[1])
            loss /= n_data
        return loss
    
    def kld_loss(self, infer, prior, mask=None):
        """KLD loss between inferred and prior z."""
        infer_mean, infer_std = infer
        prior_mean, prior_std = prior
        return losses.kld_gauss(infer_mean, infer_std,
                                prior_mean, prior_std, mask)

    def rec_loss(self, inputs, recon, mask=None, rec_mults={}):
        """Input reconstruction loss."""
        loss = 0.0
        rec_mean, rec_std = recon
        for m in self.modalities:
            if m not in inputs:
                continue
            mult = 1.0 if m not in rec_mults else rec_mults[m]
            if mult == 0:
                continue
            if self.dists[m] == 'Bernoulli':
                loss += mult * losses.nll_bernoulli(rec_mean[m],
                                                    inputs[m], mask)
            else:
                loss += mult * losses.nll_gauss(rec_mean[m], rec_std[m],
                                                inputs[m], mask)
        return loss
            
    def _sample_gauss(self, mean, std):
        """Use std to sample."""
        eps = torch.FloatTensor(std.size()).to(self.device).normal_()
        return eps.mul(std).add_(mean)
