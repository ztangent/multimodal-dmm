"""Multimodal Bidirectional Deep Markov Model (MBDMM).

Original DMM described by Krishan et. al. (https://arxiv.org/abs/1609.09869)

To handle missing modalities, we use the MVAE approach
described by Wu & Goodman (https://arxiv.org/abs/1802.05335).

Requires pytorch >= 0.4.1 for nn.ModuleDict
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import math
import torch
import torch.nn as nn

from .common import GaussianMLP, GaussianGTF
from .dgts import MultiDGTS

class MultiBDMM(MultiDGTS):
    def __init__(self, modalities, dims, encoders=None, decoders=None,
                 h_dim=32, z_dim=32, z0_mean=0.0, z0_std=1.0, min_std=1e-3,
                 learn_glb_prior=True, device=torch.device('cuda:0')):
        """
        Construct multimodal bidirectional deep Markov model.

        modalities : list of str
            list of names for each modality
        dims : list of int
            list of feature dimensions for each modality
        encoders : list of nn.Module
            list of custom encoder modules for each modality
        decoders : list of nn.Module
            list of custom decoder modules for each modality
        h_dim : int
            size of intermediary layers
        z_dim : int
            number of latent dimensions
        z0_mean : float
            mean of global latent prior
        z0_std : float
            standard deviation of global latent prior
        device : torch.device
            device on which this module is stored (CPU or GPU)
        """
        super(MultiBDMM, self).__init__()
        self.modalities = modalities
        self.n_mods = len(modalities)
        self.dims = dict(zip(modalities, dims))
        self.h_dim = h_dim
        self.z_dim = z_dim
            
        # Encoders for each modality p(z|x) = N(mu(x), sigma(x))
        self.enc = nn.ModuleDict()            
        if encoders is not None:
            # Use custom encoders if provided
            if type(encoders) is list:
                encoders = zip(modalities, encoders)
            self.enc.update(encoders)
        else:
            # Default to MLP with single-layer feature extractor
            for m in self.modalities:
                self.enc[m] = nn.Sequential(
                    nn.Linear(self.dims[m], h_dim),
                    nn.ReLU(),
                    GaussianMLP(h_dim, z_dim, h_dim))

        # Decoders for each modality p(xi|z) = N(mu(z), sigma(z))
        self.dec = nn.ModuleDict()
        if decoders is not None:
            # Use custom decoders if provided
            if type(decoders) is list:
                decoders = zip(modalities, decoders)
            self.enc.update(decoders)
        else:
            # Default to MLP
            for m in self.modalities:
                self.dec[m] = GaussianMLP(z_dim, self.dims[m], h_dim)

        # Latent state transitions p(z|z_prev) = N(mu(z_prev), sigma(z_prev))
        self.trans = nn.ModuleDict()
        self.trans['fwd'] = GaussianGTF(z_dim, h_dim, min_std=min_std)
        self.trans['bwd'] = GaussianGTF(z_dim, h_dim, min_std=min_std)

        # Global prior on latent space
        self.z0_mean = nn.Parameter(z0_mean * torch.ones(1, z_dim))
        self.z0_log_std = nn.Parameter((z0_std * torch.ones(1, z_dim)).log())
        self.min_std = min_std
        if not learn_glb_prior:
            self.z0_mean.requires_grad = False
            self.z0_log_std.requires_grad = False
        
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

        inputs : dict of str : torch.tensor
           keys are modality names, tensors are (T, B, D)
           for max sequence length T, batch size B and input dims D
           NOTE: should have at least one modality present
        combine : bool
           if true, combines inferred Gaussian distributions using
           the product of experts formula
        """
        t_max, b_dim = inputs[inputs.keys()[0]].shape[:2]
        
        # Accumulate inferred parameters for each modality
        z_mean, z_std, masks = [], [], []

        for m in self.modalities:
            # Ignore missing modalities
            if m not in inputs:
                continue
            # Mask out NaNs
            mask_m = 1 - torch.isnan(inputs[m]).any(dim=-1)
            input_m = torch.tensor(inputs[m])
            input_m[torch.isnan(input_m)] = 0.0
            # Compute mean and std of latent z given modality m
            z_mean_m, z_std_m = self.enc[m](input_m.view(-1, self.dims[m]))
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

        z : torch.tensor
           tensor of shape (T, B, D) for max sequence length T, batch size B
           and latent dims D 
        """
        t_max, b_dim = z.shape[:2]
        out_mean, out_std = dict(), dict()        
        for m in self.modalities:
            out_mean_m, out_std_m = self.dec[m](z.view(-1, self.z_dim))
            out_mean[m] = out_mean_m.reshape(t_max, b_dim, -1)
            out_std[m] = out_std_m.reshape(t_max, b_dim, -1)
        return out_mean, out_std

    def z_sample(self, t_max, b_dim, direction='fwd', sample=True):
        """Generates a sequence of latent variables."""
        z_mean, z_std = [], []
        for t in range(t_max):
            if t > 0:
                z_mean_t, z_std_t = self.trans[direction](z_t)
            else:
                z_mean_t, z_std_t, _ = self.prior((b_dim, 1))
            z_mean.append(z_mean_t)
            z_std.append(z_std_t)
            if sample:
                z_t = self._sample_gauss(z_mean_t, z_std_t)
            else:
                z_t = z_mean_t
        if direction == 'bwd':
            z_mean.reverse()
            z_std.reverse()
        z_mean, z_std = torch.stack(z_mean), torch.stack(z_std)
        return z_mean, z_std

    def z_filter(self, z_mean, z_std, z_masks, init_mask=1,
                 direction='fwd', sample=True, n_particles=1):
        """Performs filtering on the latent variables by combining
        the prior distributions with inferred distributions
        at each time step using a product of Gaussian experts.

        z_mean : torch.tensor
            (M, T, B, D) for M experts, T steps, batch size B, D latent dims
        z_std : torch.tensor
            (M, T, B, D) for M experts, T steps, batch size B, D latent dims
        z_masks : torch.tensor
            (M, T, B) for M experts, T steps, batch size B
        init_mask : int
            mask for initial time step
        direction : str
            'fwd' or 'bwd' filtering
        sample : bool
            sample at each timestep if true, use the mean otherwise
        n_particles : int
            number of filtering particles
        """
        t_max, b_dim = z_mean[0].shape[:2]
        
        # Initialize list accumulators
        prior_mean, prior_std = [], []
        infer_mean, infer_std = [], []
        samples = []

        # Reverse inputs in time if direction is backward
        rv = ( (lambda x : list(reversed(x))) if direction == 'bwd'
               else (lambda x : x) )
        # Set whether to use forward or backward transition
        trans_f = self.trans[direction]
        
        for t in rv(range(t_max)):
            # Compute prior p(z|z_prev) at time t
            prior_mask_t =\
                torch.ones((b_dim,), dtype=torch.uint8).to(self.device)
            if len(samples) == 0:
                # Use default prior
                prior_mean_t, prior_std_t, _ = self.prior((b_dim, 1))
                prior_mask_t = init_mask * prior_mask_t
            elif n_particles == 1:
                prior_mean_t, prior_std_t = trans_f(z_particles[0])
            else:
                # Compute params for each particle, then average
                prior_t = trans_f(z_particles.view(-1, self.z_dim))
                prior_mean_t = prior_t[0].view(n_particles, -1, self.z_dim)
                prior_std_t = prior_t[1].view(n_particles, -1, self.z_dim)
                prior_mean_t, prior_std_t =\
                    self.mean_of_experts(prior_mean_t, prior_std_t)
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
            if sample or n_particles > 1:
                z_particles = self._sample_gauss(
                    infer_mean_t.expand(n_particles,-1,-1),
                    infer_std_t.expand(n_particles,-1,-1))
                samples.append(z_particles.mean(dim=0))
            else:
                z_particles = infer_mean_t.unsqueeze(0)
                samples.append(infer_mean_t)

        # Concatenate outputs to tensor, reversing if necessary
        infer = (torch.stack(rv(infer_mean)), torch.stack(rv(infer_std)))
        prior = (torch.stack(rv(prior_mean)), torch.stack(rv(prior_std)))
        samples = torch.stack(rv(samples))
        
        return infer, prior, samples
            
    def sample(self, t_max, b_dim, direction='fwd'):
        """Generates a sequence of the input data by sampling."""
        z_mean, z_std = self.z_sample(t_max, b_dim, direction, sample=True)
        out_mean, out_std = self.decode(z_mean)
        return out_mean, out_std
        
    def forward(self, inputs, **kwargs):
        """Takes in (optionally missing) inputs and reconstructs them.

        inputs : dict of str : torch.tensor
           keys are modality names, tensors are (T, B, D)
           for max sequence length T, batch size B and input dims D
        lengths : list of int
           lengths of all input sequences in the batch
        mode : 'ffilter', 'bfilter', 'fsmooth' (default), 'bsmooth'
           whether to filter or smooth, and in which directions
        sample: bool
           whether to sample from z_t (default) or return MAP estimate
        """
        lengths = kwargs.get('lengths')
        mode = kwargs.get('mode', 'fsmooth')
        sample = kwargs.get('sample', True)
        flt_particles = kwargs.get('flt_particles', 1)
        smt_particles = kwargs.get('smt_particles', 1)
        t_max, b_dim = max(lengths), len(lengths)
        
        # Setup global (i.e. time-invariant) prior on z
        glb_mean, glb_std, glb_mask = self.prior((t_max, b_dim, 1))
        
        # Infer z_t from x_t without temporal information
        obs_mean, obs_std, obs_mask = self.encode(inputs)

        # Define helper function to prepend tensors
        cons = lambda x, y : torch.cat([x.unsqueeze(0), y], dim=0)
        
        # Filtering pass
        direction = 'fwd' if mode in ['ffilt', 'bsmooth'] else 'bwd'
        infer, prior, z_samples = \
            self.z_filter(cons(glb_mean, obs_mean), cons(glb_std, obs_std),
                          cons(glb_mask, obs_mask), init_mask=0,
                          direction=direction, sample=sample,
                          n_particles=flt_particles)

        # Smoothing pass
        if mode in ['fsmooth', 'bsmooth']:
            direction = 'fwd' if mode == 'fsmooth' else 'bwd'
            flt_mean, flt_std = prior
            flt_mask = torch.tensor(glb_mask)
            flt_mask[-1] = 0 * flt_mask[-1]
            infer, prior, z_samples = \
                self.z_filter(cons(glb_mean, cons(flt_mean, obs_mean)),
                              cons(glb_std, cons(flt_std, obs_std)),
                              cons(glb_mask, cons(flt_mask, obs_mask)),
                              init_mask=1, direction=direction, sample=sample,
                              n_particles=smt_particles)

        # Decode sampled z to reconstruct inputs
        out_mean, out_std = self.decode(z_samples)
        outputs = (out_mean, out_std)

        return infer, prior, outputs

    def kld_prior(self, n_particles, direction='fwd'):
        """Compute KL divergence between E[p(z_next|z)] and p(z)."""
        cur_mean, cur_std, _ = self.prior((n_particles, 1))
        samples = self._sample_gauss(cur_mean, cur_std)
        nxt_mean, nxt_std = self.trans[direction](samples)
        nxt_mean, nxt_std = self.mean_of_experts(nxt_mean, nxt_std)
        cur_mean, cur_std, _ = self.prior((1, 1))
        loss = self._kld_gauss(cur_mean, cur_std, nxt_mean, nxt_std)
        return loss
    
    def step(self, inputs, mask, kld_mult, rec_mults, targets=None, **kwargs):
        """Custom training step for bidirectional training paradigm."""
        # Set up arguments
        f_mode = kwargs.get('f_mode', 'bfilt')
        s_mode = kwargs.get('s_mode', 'fsmooth')
        f_mult, s_mult = kwargs.get('f_mult', 0.5), kwargs.get('s_mult', 0.5)
        t_max, b_dim = mask.shape[:2]
        
        loss = 0
        # Compute prior matching loss
        loss += kld_mult * t_max * self.kld_prior(b_dim, 'fwd')
        loss += kld_mult * t_max * self.kld_prior(b_dim, 'bwd')
        # Compute loss when backward filtering
        loss += f_mult * super(MultiBDMM, self).\
            step(inputs, mask, kld_mult, rec_mults, mode=f_mode, **kwargs)
        # Compute loss when forward smoothing
        loss += s_mult * super(MultiBDMM, self).\
            step(inputs, mask, kld_mult, rec_mults, mode=s_mode, **kwargs)
        return loss
    
if __name__ == "__main__":
    # Test code by running 'python -m models.bdmm' from base directory
    import os, sys, argparse
    from datasets.spirals import SpiralsDataset
    from datasets.multiseq import seq_collate_dict

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default="../../data",
                        help='data directory')
    parser.add_argument('--subset', type=str, default="train",
                        help='whether to load train/test data')
    args = parser.parse_args()

    print("Loading data...")
    dataset = SpiralsDataset(['spiral-x', 'spiral-y'],
                             args.dir, args.subset, base_rate=2.0,
                             truncate=True, item_as_dict=True)
    print("Building model...")
    model = MultiBDMM(['spiral-x', 'spiral-y'], [1, 1],
                      device=torch.device('cpu'))
    model.eval()
    print("Passing a sample through the model...")
    data, mask, lengths = seq_collate_dict([dataset[0]])
    infer, prior, outputs = model(data, lengths=lengths)
    out_mean, out_std = outputs
    print("Predicted:")
    for x, y in zip(out_mean['spiral-x'], out_mean['spiral-y']):
        print("{:+0.3f}, {:+0.3f}".format(x.item(), y.item()))
