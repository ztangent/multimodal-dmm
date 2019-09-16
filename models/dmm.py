"""Multimodal Deep Markov Model (MDMM).

Original DMM described by Krishnan et. al. (https://arxiv.org/abs/1609.09869)

To handle missing data, we use the MVAE approach
described by Wu & Goodman (https://arxiv.org/abs/1802.05335),
combined with a varational backward-forward pass.

We call this inference technique bi-directional factorized
variational inference (BFVI).

Requires pytorch >= 0.4.1 for nn.ModuleDict
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from builtins import zip, range

import numpy as np
import torch
import torch.nn as nn

from . import common, losses
from .dgts import MultiDGTS

class MultiDMM(MultiDGTS):
    def __init__(self, modalities, dims, dists=None,
                 encoders=None, decoders=None, h_dim=32, z_dim=32,
                 z0_mean=0.0, z0_std=1.0, min_std=1e-3,
                 device=torch.device('cuda:0')):
        """
        Construct multimodal deep Markov model.

        Parameters
        ----------
        modalities : list of str
            list of names for each modality
        dims : list of int or tuple
            list of feature dimensions / dims for each modality
        dists : list of {'Normal', 'Bernoulli', 'Categorical'}
            list of distributions for each modality
        encoders : list or dict of nn.Module
            list or dict of custom encoder modules for each modality
        decoders : list or dict of nn.Module
            list or dict of custom decoder modules for each modality
        h_dim : int
            size of intermediary layers
        z_dim : int
            number of latent dimensions
        z0_mean : float
            mean of global latent prior
        z0_std : float
            standard deviation of global latent prior
        min_std : float
            minimum std to ensure stable training
        device : torch.device
            device on which this module is stored (CPU or GPU)
        """
        super(MultiDMM, self).__init__()
        self.modalities = modalities
        self.n_mods = len(modalities)
        self.dims = dict(zip(modalities, dims))
        self.h_dim = h_dim
        self.z_dim = z_dim

        # Default to Gaussian distributions
        if dists is None:
            dists = ['Normal'] * self.n_mods
        self.dists = dict(zip(modalities, dists))

        # Encoders for each modality q'(z|x) = N(mu(x), sigma(x))
        # Where q'(z|x) = p(z|x) / p(z)
        self.enc = nn.ModuleDict()
        # Default to MLP encoder (assumes 1D input)
        for m in self.modalities:
            if self.dists[m] == 'Categorical':
                self.enc[m] = nn.Sequential(
                    nn.Embedding(np.prod(self.dims[m]), h_dim),
                    nn.ReLU(),
                    common.GaussianMLP(h_dim, z_dim, h_dim))
            else:
                self.enc[m] = common.GaussianMLP(
                    np.prod(self.dims[m]), z_dim, h_dim)
        if encoders is not None:
            # Use custom encoders if provided
            if type(encoders) is list:
                encoders = list(zip(modalities, encoders))
            self.enc.update(encoders)

        # Decoders for each modality p(x|z) = N(mu(z), sigma(z))
        self.dec = nn.ModuleDict()
        # Default to MLP with 1D output
        for m in self.modalities:
            if self.dists[m] == 'Categorical':
                self.dec[m] = common.CategoricalMLP(
                    z_dim, np.prod(self.dims[m]), h_dim)
            else:
                self.dec[m] = common.GaussianMLP(
                    z_dim, np.prod(self.dims[m]), h_dim)
        if decoders is not None:
            # Use custom decoders if provided
            if type(decoders) is list:
                decoders = list(zip(modalities, decoders))
            self.dec.update(decoders)

        # State transitions q'(z|z_prev) = N(mu(z_prev), sigma(z_prev))
        # Where q'(z|z_prev) = p(z|z_prev) / p(z)
        self.trans = nn.ModuleDict()
        self.trans['fwd'] = common.GaussianGTF(z_dim, h_dim, min_std=min_std)
        self.trans['bwd'] = common.GaussianGTF(z_dim, h_dim, min_std=min_std)

        # Global prior on latent space
        self.z0_mean = nn.Parameter(z0_mean * torch.ones(1, z_dim))
        self.z0_log_std = nn.Parameter((z0_std * torch.ones(1, z_dim)).log())
        self.min_std = min_std

        # Store module in specified device (CUDA/CPU)
        self.device = (device if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.to(self.device)

    def prior(self, shape):
        """Returns params of global prior in specified shape."""
        mean = self.z0_mean.repeat(*shape)
        std = (self.z0_log_std.exp() + self.min_std).repeat(*shape)
        mask = torch.ones(shape[:-1], dtype=torch.uint8).to(self.device)
        return mean, std, mask

    def encode(self, inputs, combine=False):
        """Encode (optionally missing) inputs to latent space.

        Parameters
        ----------
        inputs : dict of str : torch.tensor
           keys are modality names, tensors are (T, B, D)
           for max sequence length T, batch size B and input dims D
           NOTE: should have at least one modality present
        combine : bool
           if true, combines inferred Gaussian distributions using
           the product of experts formula

        Returns
        -------
        z_mean : torch.tensor
            inferred latent mean for each modality
            shape is (M, T, B, D) if combine is False, otherwise (T, B, D)
        z_std : torch.tensor
            inferred latent std for each modality, same shape as z_mean
        masks : torch.tensor
            masks for batches and timepoints with missing inputs
            shape is (M, T, B) if combine is False, otherwise (T, B)
        """
        t_max, b_dim = inputs[list(inputs.keys())[0]].shape[:2]

        # Accumulate inferred parameters for each modality
        z_mean, z_std, masks = [], [], []

        for m in self.modalities:
            # Ignore missing modalities
            if m not in inputs:
                continue
            # Mask out all timesteps with NaNs
            mask_m = 1 - torch.isnan(inputs[m]).flatten(2,-1).any(dim=-1)
            input_m = inputs[m].clone().detach()
            input_m[torch.isnan(input_m)] = 0.0
            if self.dists[m] == 'Categorical':
                input_m = input_m.long()
            # Compute mean and std of latent z given modality m
            z_mean_m, z_std_m = self.enc[m](input_m.flatten(0,1))
            z_mean_m = z_mean_m.reshape(t_max, b_dim, -1)
            z_std_m = z_std_m.reshape(t_max, b_dim, -1)
            # Add p(z|x_m) to the PoE calculation
            z_mean.append(z_mean_m)
            z_std.append(z_std_m)
            masks.append(mask_m)

        z_mean = torch.stack(z_mean, dim=0)
        z_std = torch.stack(z_std, dim=0)
        masks = torch.stack(masks, dim=0)

        if combine:
            # Combine the Gaussian parameters using PoE
            z_mean, z_std = \
                self.product_of_experts(z_mean, z_std, masks)
            # Compute OR of masks across modalities
            masks = masks.any(dim=0)

        return z_mean, z_std, masks

    def decode(self, z):
        """Decode from latent space to inputs.

        Parameters
        ----------
        z : torch.tensor
           shape is (T, B, D) for T timesteps, B batch dims, D latent dims

        Returns
        -------
        recon : dict of str : (torch.tensor, ...)
           tuple of reconstructed distribution parameters for each modality
        """
        t_max, b_dim = z.shape[:2]
        recon = dict()
        for m in self.modalities:
            recon_m = self.dec[m](z.view(-1, self.z_dim))
            rec_shape = [t_max, b_dim] + list(recon_m[0].shape[1:])
            # Reshape each output parameter (e.g. mean, std) to (T, B, ...)
            recon[m] = tuple(r.reshape(*rec_shape) for r in recon_m)
        return recon

    def z_next(self, z, direction='fwd', glb_prior=None):
        """Compute p(z_next|z) given a tensor of z particles

        Parameters
        ----------
        z : torch.tensor
           shape is (K, B, D) for K particles, B batch dims, D latent dims
        direction : {'fwd', 'bwd'}
           which direction to compute
        glb_prior : (torch.tensor, torch.tensor)
           optionally provide prior parameters to reuse

        Returns
        -------
        z_mean : torch.tensor
            mean of z_next, shape is (B, D)
        z_std : torch.tensor
            std of z_next, shape is (B, D)
        """

        if glb_prior is None:
            glb_mean, glb_std, _ = self.prior(z.shape[1:])
        else:
            glb_mean, glb_std = glb_prior

        if z.shape[0] == 1:
            # Compute p(z|z_prev) = p(z) * q'(z|z_prev)
            q_mean, q_std = self.trans[direction](z[0])
            z_mean = torch.stack([glb_mean, q_mean])
            z_std = torch.stack([glb_std, q_std])
            z_mean, z_std = self.product_of_experts(z_mean, z_std)

            return z_mean, z_std

        # Compute p(z|z_prev) for each particle
        q_mean, q_std = self.trans[direction](z.view(-1, self.z_dim))
        z_mean = torch.stack([glb_mean.repeat(z.shape[0], 1), q_mean])
        z_std = torch.stack([glb_std.repeat(z.shape[0], 1), q_std])
        z_mean, z_std = self.product_of_experts(z_mean, z_std)

        # Reshape and average across particles
        z_mean, z_std = z_mean.view(*z.shape), z_std.view(*z.shape)
        z_mean, z_std = self.mean_of_experts(z_mean, z_std)

        return z_mean, z_std

    def z_sample(self, t_max, b_dim, direction='fwd',
                 sample=True, n_particles=1, z_init=None, inclusive=False):
        """Generates a sequence of latent variables.

        Parameters
        ----------
        t_max : int
            number of timesteps T to sample
        b_dim : int
            batch size B
        direction : 'fwd' or 'bwd'
            whether to sample forwards or backwards in time
        sample : bool
            whether to sample (default) or to use MAP estimate
        n_particles : int
            number of sampling particles, overrides sample flag if > 1
        z_init : (torch.tensor, torch.tensor)
            (mean, std) of initial latent distribution (default: global prior)
        inclusive : bool
            flag to include initial state in returned tensor (default : False)

        Returns
        -------
        z_mean : torch.tensor
            mean of z_next, shape is (T, B, D) for D latent dims
        z_std : torch.tensor
            std of z_next, shape is (T, B, D) for D latent dims
        """
        glb_mean, glb_std, _ = self.prior((b_dim, 1))
        z_mean, z_std = [], []

        # Initialize latent distribution
        z_mean_t, z_std_t = glb_mean, glb_std if z_init is None else z_init
        if inclusive:
            z_mean.append(z_mean_t)
            z_std.append(z_std_t)

        for t in range(t_max - int(inclusive)):
            # Sample particles for current timestep
            if sample or n_particles > 1:
                z_t = self._sample_gauss(
                    z_mean_t.expand(n_particles, -1, -1),
                    z_std_t.expand(n_particles, -1, -1))
            else:
                z_t = z_mean_t.unsqueeze(0)

            # Compute parameters for next time step
            z_mean_t, z_std_t =\
                self.z_next(z_t, direction, (glb_mean, glb_std))
            z_mean.append(z_mean_t)
            z_std.append(z_std_t)

        if direction == 'bwd':
            z_mean.reverse()
            z_std.reverse()
        z_mean, z_std = torch.stack(z_mean), torch.stack(z_std)

        return z_mean, z_std

    def z_filter(self, z_mean, z_std, z_masks, direction='fwd',
                 sample=True, n_particles=1, sample_init=False):
        """Performs filtering on the latent variables by combining
        the prior distributions with inferred distributions
        at each time step using a product of Gaussian experts.

        Parameters
        ----------
        z_mean : torch.tensor
            means of inferred distributions to combine
            (M, T, B, D) for M experts, T steps, batch size B, D latent dims
        z_std : torch.tensor
            std of inferred distributions to combine, same shape as z_mean
        z_masks : torch.tensor
            masks away contributions of inferred distributions, shape (M, T, B)
        direction : {'fwd', 'bwd'}
            'fwd' or 'bwd' filtering
        sample : bool
            sample at each timestep if true, use the mean otherwise
        n_particles : int
            number of filtering particles, overrides sample flag if > 1
        sample_init : bool
            whether to sample for initial time-step

        Returns
        -------
        infer : (torch.tensor, torch.tensor)
            (mean, std) of time-wise cond. posteriors, shape is (T, B, D)
            when n_particles=1, corresponds to
            q(z_t|z_{t+1}, x_t) for backward filtering
            q(z_t|z_{t-1}, x_{t:T}) for forward smoothing
        prior : (torch.tensor, torch.tensor)
            (mean, std) of time-wise cond. priors, shape is (T, B, D)
            when n_particles=1, corresponds to
            p(z_t|z_{t+1}) for backward filtering
            p(z_t|z_{t-1}) for forward smoothing
        samples : torch.tensor
            z samples from time-wise cond. posterior, shope is (T, B, D)
            returns mean for each time step when n_particles > 1 are sampled
        """
        t_max, b_dim = z_mean[0].shape[:2]

        # Initialize list accumulators
        prior_mean, prior_std = [], []
        infer_mean, infer_std = [], []
        samples = []

        # Reverse inputs in time if direction is backward
        rv = ( (lambda x : list(reversed(x))) if direction == 'bwd'
               else (lambda x : x) )

        # Setup global (i.e. time-invariant) prior on z
        glb_mean, glb_std, _ = self.prior((b_dim, 1))

        for t in rv(list(range(t_max))):
            prior_mask_t =\
                torch.ones((b_dim,), dtype=torch.uint8).to(self.device)
            if len(samples) == 0:
                # Use global prior p(z) at t = 0 or t = t_max
                prior_mean_t, prior_std_t = glb_mean, glb_std
            else:
                # Compute prior p(z|z_prev) at time t
                prior_mean_t, prior_std_t =\
                    self.z_next(z_t, direction, (glb_mean, glb_std))
            prior_mean.append(prior_mean_t)
            prior_std.append(prior_std_t)

            # Concatenate means and standard deviations
            z_mean_t = torch.cat([prior_mean_t.unsqueeze(0), z_mean[:,t]], 0)
            z_std_t = torch.cat([prior_std_t.unsqueeze(0), z_std[:,t]], 0)
            masks = torch.cat([prior_mask_t.unsqueeze(0), z_masks[:,t]], 0)

            # Combine distributions using product of experts
            infer_mean_t, infer_std_t = \
                self.product_of_experts(z_mean_t, z_std_t, masks)
            infer_mean.append(infer_mean_t)
            infer_std.append(infer_std_t)

            # Sample particles from inferred distribution
            if sample or n_particles > 1 or (len(samples)==0 and sample_init):
                z_t = self._sample_gauss(
                    infer_mean_t.expand(n_particles, -1, -1),
                    infer_std_t.expand(n_particles, -1, -1))
                samples.append(z_t.mean(dim=0))
            else:
                z_t = infer_mean_t.unsqueeze(0)
                samples.append(infer_mean_t)

        # Concatenate outputs to tensor, reversing if necessary
        infer = (torch.stack(rv(infer_mean)), torch.stack(rv(infer_std)))
        prior = (torch.stack(rv(prior_mean)), torch.stack(rv(prior_std)))
        samples = torch.stack(rv(samples))

        return infer, prior, samples

    def sample(self, t_max, b_dim, direction='fwd'):
        """Generates a sequence of the input data by sampling."""
        z_mean, z_std = self.z_sample(t_max, b_dim, direction, sample=True)
        recon = self.decode(z_mean)
        return recon

    def forward(self, inputs, **kwargs):
        """Takes in (optionally missing) inputs and reconstructs them.

        Parameters
        ----------
        inputs : dict of str : torch.tensor
           keys are modality names, tensors are (T, B, D, ...)
           for max sequence length T, batch size B and input dims (D, ...)
        lengths : list of int
           lengths of all input sequences in the batch
        mode : {'fsmooth', 'bsmooth', 'ffilter', 'bfilter'}
           whether to filter or smooth, and in which directions
        sample: bool
           whether to sample from z_t (default) or return MAP estimate
        sample_init: bool
           whether to sample from z_0 or use mean (default)
        flt_particles : int
           number of filtering particles (default : 1)
        smt_particles : int
           number of smoothing particles (default : 1)

        Returns
        -------
        infer : (torch.tensor, torch.tensor)
            (mean, std) of time-wise cond. posteriors, see z_filter
        prior : (torch.tensor, torch.tensor)
            (mean, std) of time-wise cond. priors, see z_filter
        recon : dict of str : (torch.tensor, ...)
           tuple of reconstructed distribution parameters for each modality
        """
        lengths = kwargs.get('lengths')
        mode = kwargs.get('mode', 'fsmooth')
        sample = kwargs.get('sample', True)
        sample_init = kwargs.get('sample_init', False)
        flt_particles = kwargs.get('flt_particles', 1)
        smt_particles = kwargs.get('smt_particles', 1)
        t_max, b_dim = max(lengths), len(lengths)

        # Helper function to append tuple of 3D tensors to 4D tensor
        cons = lambda a, b : torch.stack(a.unbind(0) + b, dim=0)

        # Infer z_t from x_t without temporal information
        obs_mean, obs_std, obs_mask = self.encode(inputs)

        # Filtering pass
        direction = 'fwd' if mode in ['ffilter', 'bsmooth'] else 'bwd'
        flt_init = sample_init if mode in ['ffilter', 'bfilter'] else False
        infer, prior, z_samples = \
            self.z_filter(obs_mean, obs_std, obs_mask, direction=direction,
                          sample=sample, n_particles=flt_particles,
                          sample_init=flt_init)

        # Smoothing pass
        if mode in ['fsmooth', 'bsmooth']:
            direction = 'fwd' if mode == 'fsmooth' else 'bwd'
            # Introduce [p(z_t)]^-1 into the product of Gaussians
            inv_mean, inv_std, inv_mask = self.prior((t_max, b_dim, 1))
            inv_std = -inv_std
            # Collect p(z_t|x_{t+1:T}) from output of filtering pass
            flt_mean, flt_std = prior
            flt_mask =\
                torch.ones((t_max, b_dim), dtype=torch.uint8).to(self.device)
            flt_mask[-1] = 0 * flt_mask[-1]
            infer, prior, z_samples = \
                self.z_filter(cons(obs_mean, (flt_mean, inv_mean)),
                              cons(obs_std, (flt_std, inv_std)),
                              cons(obs_mask, (flt_mask, inv_mask)),
                              direction=direction,
                              sample=sample, n_particles=smt_particles,
                              sample_init=sample_init)

        # Decode sampled z to reconstruct inputs
        recon = self.decode(z_samples)

        return infer, prior, recon

    def kld_prior(self, n_particles, direction='fwd'):
        """Compute KL divergence between E[p(z_next|z)] and p(z)."""
        glb_mean, glb_std, _ = self.prior((1, 1, 1))
        nxt_mean, nxt_std = self.z_sample(1, 1, direction, True, n_particles)
        loss = losses.kld_gauss(glb_mean, glb_std, nxt_mean, nxt_std)
        return loss

    def step(self, inputs, mask, kld_mult, rec_mults,
             targets=None, uni_loss=True, **kwargs):
        """Custom training step for bidirectional training paradigm.
        See :func:`~models.MultiDGTS.step` for non-keyword arguments

        Parameters
        ----------
        f_mode : {'bfilter', 'ffilter'}
            mode when computing filtering loss
        s_mode : {'fsmooth', 'bsmooth'}
            mode when computing smoothing loss
        f_mult : float
            how much to weight filtering loss (default : 0.5)
        s_mult : float
            how much to weight smoothing loss (default : 0.5)
        match_mult : float
            how much to weight prior matching loss (default : 0.01)
        train_particles : int
            n_particles for filtering pass when smoothing (default : 25)
        match_particles : int
            n_particles for prior matching computation

        Returns
        -------
        loss : torch.tensor
            total training loss for this step
        """
        # Extract arguments
        f_mode = kwargs.get('f_mode', 'bfilter')
        s_mode = kwargs.get('s_mode', 'fsmooth')
        f_mult, s_mult = kwargs.get('f_mult', 0.5), kwargs.get('s_mult', 0.5)
        match_mult = kwargs.get('match_mult', 0.01)
        train_particles = kwargs.get('train_particles', 25)
        match_particles = kwargs.get('match_particles', 50)
        t_max, b_dim = mask.shape[:2]

        loss = 0
        # Compute prior matching loss
        if match_mult > 0:
            loss += (match_mult * kld_mult * mask.sum().float() *
                     self.kld_prior(match_particles, 'fwd'))
            loss += (match_mult * kld_mult * mask.sum().float() *
                     self.kld_prior(match_particles, 'bwd'))
        # Compute loss when filtering
        loss += f_mult * super(MultiDMM, self).\
            step(inputs, mask, kld_mult, rec_mults, targets, uni_loss,
                 mode=f_mode, **kwargs)
        # Compute loss when smoothing
        loss += s_mult * super(MultiDMM, self).\
            step(inputs, mask, kld_mult, rec_mults, targets, uni_loss,
                 mode=s_mode, flt_particles=train_particles, **kwargs)
        return loss

if __name__ == "__main__":
    # Test code by running 'python -m models.bdmm' from base directory
    import os, sys, argparse
    from datasets.spirals import SpiralsDataset
    from datasets.multiseq import seq_collate_dict

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default="./data",
                        help='data directory')
    parser.add_argument('--subset', type=str, default="train",
                        help='whether to load train/test data')
    args = parser.parse_args()

    print("Loading data...")
    dataset = SpiralsDataset(['spiral-x', 'spiral-y'],
                             args.dir, args.subset,
                             truncate=True, item_as_dict=True)
    print("Building model...")
    model = MultiDMM(['spiral-x', 'spiral-y'], [1, 1],
                      device=torch.device('cpu'))
    model.eval()
    print("Passing a sample through the model...")
    data, mask, lengths, order = seq_collate_dict([dataset[0]])
    infer, prior, recon = model(data, lengths=lengths)
    print("Predicted:")
    for x, y in zip(recon['spiral-x'][0], recon['spiral-y'][0]):
        print("{:+0.3f}, {:+0.3f}".format(x.item(), y.item()))
