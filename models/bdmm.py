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
        min_std : float
            minimum std to ensure stable training
        device : torch.device
            device on which this module is stored (CPU or GPU)
        """
        super(MultiBDMM, self).__init__()
        self.modalities = modalities
        self.n_mods = len(modalities)
        self.dims = dict(zip(modalities, dims))
        self.h_dim = h_dim
        self.z_dim = z_dim
            
        # Encoders for each modality q'(z|x) = N(mu(x), sigma(x))
        # Where q'(z|x) = p(z|x) / p(z) 
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

        # Decoders for each modality p(x|z) = N(mu(z), sigma(z))
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

        # State transitions q'(z|z_prev) = N(mu(z_prev), sigma(z_prev))
        # Where q'(z|z_prev) = p(z|z_prev) / p(z) 
        self.trans = nn.ModuleDict()
        self.trans['fwd'] = GaussianGTF(z_dim, h_dim, min_std=min_std)
        self.trans['bwd'] = GaussianGTF(z_dim, h_dim, min_std=min_std)

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
           shape is (T, B, D) for T timesteps, B batch dims, D latent dims
        """
        t_max, b_dim = z.shape[:2]
        out_mean, out_std = dict(), dict()        
        for m in self.modalities:
            out_mean_m, out_std_m = self.dec[m](z.view(-1, self.z_dim))
            out_mean[m] = out_mean_m.reshape(t_max, b_dim, -1)
            out_std[m] = out_std_m.reshape(t_max, b_dim, -1)
        return out_mean, out_std

    def z_next(self, z, direction='fwd', glb_prior=None):
        """Compute p(z_next|z) given z

        z : torch.tensor
           shape is (K, B, D) for K particles, B batch dims, D latent dims
        direction : 'fwd' or 'bwd'
           which direction to compute
        glb_prior : (torch.tensor, torch.tensor)
           optionally provide prior parameters to reuse
        """

        if glb_prior is None:
            glb_mean, glb_std, _ = self.prior(z.shape[1:])
        else:
            glb_mean, glb_std = glb_prior
        
        if z.shape[0] == 1:
            # Compute p(z|z_prev) = p(z) * q'(z|z_prev)
            q_mean_t, q_std_t = self.trans[direction](z[0])
            z_mean_t = torch.stack([glb_mean, q_mean_t])
            z_std_t = torch.stack([glb_std, q_std_t])
            z_mean_t, z_std_t = self.product_of_experts(z_mean_t, z_std_t)
            
            return z_mean_t, z_std_t

        # Compute p(z|z_prev) for each particle
        q_mean_t, q_std_t = self.trans[direction](z.view(-1, self.z_dim))
        z_mean_t = torch.stack([glb_mean.repeat(z.shape[0], 1), q_mean_t])
        z_std_t = torch.stack([glb_std.repeat(z.shape[0], 1), q_std_t])
        z_mean_t, z_std_t = self.product_of_experts(z_mean_t, z_std_t)

        # Reshape and average across particles
        z_mean_t, z_std_t = z_mean_t.view(*z.shape), z_std_t.view(*z.shape)
        z_mean_t, z_std_t = self.mean_of_experts(z_mean_t, z_std_t)

        return z_mean_t, z_std_t
        
    def z_sample(self, t_max, b_dim, direction='fwd',
                 sample=True, n_particles=1, z_init=None, inclusive=False):
        """Generates a sequence of latent variables.

        t_max : int
            number of timesteps to sample
        b_dim : int
            batch size
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
                 sample=True, n_particles=1):
        """Performs filtering on the latent variables by combining
        the prior distributions with inferred distributions
        at each time step using a product of Gaussian experts.

        z_mean : torch.tensor
            (M, T, B, D) for M experts, T steps, batch size B, D latent dims
        z_std : torch.tensor
            (M, T, B, D) for M experts, T steps, batch size B, D latent dims
        z_masks : torch.tensor
            (M, T, B) for M experts, T steps, batch size B
        direction : str
            'fwd' or 'bwd' filtering
        sample : bool
            sample at each timestep if true, use the mean otherwise
        n_particles : int
            number of filtering particles, overrides sample flag if > 1
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
        
        for t in rv(range(t_max)):
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
            if sample or n_particles > 1:
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
        flt_particles : int
           number of filtering particles (default : 1)
        smt_particles : int
           number of smoothing particles (default : 1)
        """
        lengths = kwargs.get('lengths')
        mode = kwargs.get('mode', 'fsmooth')
        sample = kwargs.get('sample', True)
        flt_particles = kwargs.get('flt_particles', 1)
        smt_particles = kwargs.get('smt_particles', 1)
        t_max, b_dim = max(lengths), len(lengths)

        # Define helper function to prepend tensors
        cons = lambda x, y : torch.cat([x.unsqueeze(0), y], dim=0)
        
        # Infer z_t from x_t without temporal information
        obs_mean, obs_std, obs_mask = self.encode(inputs)
        
        # Filtering pass
        direction = 'fwd' if mode in ['ffilt', 'bsmooth'] else 'bwd'
        infer, prior, z_samples = \
            self.z_filter(obs_mean, obs_std, obs_mask, direction=direction,
                          sample=sample, n_particles=flt_particles)

        # Smoothing pass
        if mode in ['fsmooth', 'bsmooth']:
            direction = 'fwd' if mode == 'fsmooth' else 'bwd'
            flt_mean, flt_std = prior
            flt_mask =\
                torch.ones((t_max, b_dim), dtype=torch.uint8).to(self.device)
            flt_mask[-1] = 0 * flt_mask[-1]
            infer, prior, z_samples = \
                self.z_filter(cons(flt_mean, obs_mean), cons(flt_std, obs_std),
                              cons(flt_mask, obs_mask), direction=direction,
                              sample=sample, n_particles=smt_particles)

        # Decode sampled z to reconstruct inputs
        out_mean, out_std = self.decode(z_samples)
        outputs = (out_mean, out_std)

        return infer, prior, outputs

    def kld_prior(self, n_particles, direction='fwd'):
        """Compute KL divergence between E[p(z_next|z)] and p(z)."""
        glb_mean, glb_std, _ = self.prior((1, 1, 1))
        nxt_mean, nxt_std = self.z_sample(1, 1, direction, True, n_particles)
        loss = self._kld_gauss(glb_mean, glb_std, nxt_mean, nxt_std)
        return loss
    
    def step(self, inputs, mask, kld_mult, rec_mults, targets=None, **kwargs):
        """Custom training step for bidirectional training paradigm.
        Additional keyword arguments:

        f_mode : 'ffilt' or 'bfilt' (default)
            mode when computing filtering loss
        s_mode : 'fsmooth' (default) or 'bsmooth'
            mode when computing smoothing loss
        f_mult : float
            how much to weight filtering loss (default : 0.5)
        s_mult : float
            how much to weight smoothing loss (default : 0.5)
        """
        # Extract arguments
        f_mode = kwargs.get('f_mode', 'bfilt')
        s_mode = kwargs.get('s_mode', 'fsmooth')
        f_mult, s_mult = kwargs.get('f_mult', 0.5), kwargs.get('s_mult', 0.5)
        match_mult = kwargs.get('match_mult', 0.0)
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
        loss += f_mult * super(MultiBDMM, self).\
            step(inputs, mask, kld_mult, rec_mults, mode=f_mode, **kwargs)
        # Compute loss when smoothing
        loss += s_mult * super(MultiBDMM, self).\
            step(inputs, mask, kld_mult, rec_mults, mode=s_mode,
                 flt_particles=train_particles, **kwargs)
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
