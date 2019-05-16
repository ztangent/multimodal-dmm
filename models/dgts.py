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

        Parameters
        ----------         
        mean : torch.tensor
            (M, B, D) for M experts, batch size B, and D latent dims OR
            (M, T, B, D) with an optional time dimension
        std : torch.tensor
            (M, B, D) for M experts, batch size B, and D latent dims OR
            (M, T, B, D) with an optional time dimension
        mask : torch.tensor
            (M, B) for M experts and batch size B
            (M, T, B) with an optional time dimension

        Returns
        -------
        product_mean : torch.tensor
            mean of product Gaussian, shape (T, B, D) or (B, D)
        product_std : torch.tensor
            std of product Gaussian, shape (T, B, D) or (B, D)
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

        Parameters
        ----------
        Return mean and standard deviation of a mixture of Gaussian experts

        mean : torch.tensor
            (M, B, D) for M experts, batch size B, and D latent dims
        std : torch.tensor
            (M, B, D) for M experts, batch size B, and D latent dims
        mask : torch.tensor
            (M, B) for M experts and batch size B

        Returns
        -------
        sum_mean : torch.tensor
            mean of Gaussian mixture, shape (T, B, D) or (B, D)
        sum_std : torch.tensor
            std of Gaussian mixture, shape (T, B, D) or (B, D)
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

    def step(self, inputs, mask, kld_mult, rec_mults,
             targets=None, uni_loss=True, **kwargs):
        """Custom training step for multimodal training paradigm.

        Parameters
        ----------         
        inputs : dict of str : torch.tensor
           keys are modality names, input tensors are (T, B, D, ...)
           for max sequence length T, batch size B and input dims D
        mask : torch.tensor
           mask for batch of sequences with unequal lengths
        kld_mult : float
           how much to weight KLD loss between posterior and prior
        rec_mults: dict of str : float
           how much to weight the reconstruction loss for each modality
        targets : dict of str : torch.tensor
           optionally provide target inputs to score against,
           otherwise targets is a copy of inputs by default
        uni_loss : bool
           flag to compute ELBO for each modality on its own (default : True)

        Returns
        -------
        loss : torch.tensor
            total training loss for this step
        """
        #
        # Get rid of unrecognized modalities
        inputs = {m : inputs[m] for m in inputs if m in self.modalities}
        # If targets not provided, assume inputs are targets
        if targets == None:
            targets = inputs
        loss = 0
        # Compute negative ELBO loss for all modalities
        if len(self.modalities) > 1:
            infer, prior, recon = self.forward(inputs, **kwargs)
            loss += self.loss(targets, infer, prior, recon, mask,
                              kld_mult, rec_mults)
        if not uni_loss:
            return loss
        # Compute negative ELBO loss for individual modalities
        for m in self.modalities:
            infer, prior, recon = self.forward({m : inputs[m]}, **kwargs)
            loss += self.loss({m : targets[m]}, infer, prior, recon, mask,
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
        for m in self.modalities:
            if m not in inputs:
                continue
            mult = 1.0 if m not in rec_mults else rec_mults[m]
            if mult == 0:
                continue
            if self.dists[m] == 'Bernoulli':
                rec_prob = recon[m][0]
                loss += mult * losses.nll_bernoulli(rec_prob,
                                                    inputs[m], mask)
            elif self.dists[m] == 'Categorical':
                rec_probs = recon[m][0]
                loss += mult * losses.nll_categorical(rec_probs,
                                                      inputs[m], mask)
            elif self.dists[m] == 'Normal':
                rec_mean, rec_std = recon[m]
                loss += mult * losses.nll_gauss(rec_mean, rec_std,
                                                inputs[m], mask)
        return loss
            
    def _sample_gauss(self, mean, std):
        """Use std to sample."""
        eps = torch.FloatTensor(std.size()).to(self.device).normal_()
        return eps.mul(std).add_(mean)
