"""Multimodal Deep Markov Model (MDMM).

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

from .common import GaussianMLP
from .dgts import MultiDGTS

class MultiDMM(MultiDGTS):
    def __init__(self, modalities, dims, encoders=None, decoders=None,
                 h_dim=32, z_dim=32, z0_mean=0.0, z0_std=1.0,
                 n_bwd_particles=1, device=torch.device('cuda:0')):
        """
        Construct multimodal deep Markov model.

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
            mean of initial latent prior
        z0_std : float
            standard deviation of initial latent prior
        device : torch.device
            device on which this module is stored (CPU or GPU)
        """
        super(MultiDMM, self).__init__()
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
            
        # Forward conditional p(z|z_prev) = N(mu(z_prev), sigma(z_prev))
        self.fwd = GaussianMLP(z_dim, z_dim, h_dim)

        # Backward conditional q(z|z_next) = N(mu(z_next), sigma(z_next))
        self.bwd = GaussianMLP(z_dim, z_dim, h_dim)

        # Number of sampling particles in backward pass
        self.n_bwd_particles = n_bwd_particles
        
        # Store module in specified device (CUDA/CPU)
        self.device = (device if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.to(self.device)

        # Initial prior
        self.z0_mean = z0_mean * torch.ones(1, z_dim).to(self.device)
        self.z0_std = z0_std * torch.ones(1, z_dim).to(self.device)        

    def encode(self, inputs):
        """Encode (optionally missing) inputs to latent space.

        inputs : dict of str : torch.tensor
           keys are modality names, tensors are (T, B, D)
           for max sequence length T, batch size B and input dims D
           NOTE: should have at least one modality present
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

        # Combine the Gaussian parameters using PoE
        z_mean = torch.stack(z_mean, dim=0)
        z_std = torch.stack(z_std, dim=0)        
        masks = torch.stack(masks, dim=0)
        z_mean, z_var = \
            self.product_of_experts(z_mean, z_std.pow(2), masks)
        z_std = z_var.pow(0.5)
        
        # Compute OR of masks across modalities
        mask = masks.any(dim=0)
        
        return z_mean, z_std, mask

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

    def generate(self, t_max, b_dim, sample=True):
        """Generates a sequence of latent variables."""
        z_mean, z_std = [], []
        for t in range(t_max):
            if t > 0:
                z_mean_t, z_std_t = self.fwd(z_t)
            else:
                z_mean_t = self.z0_mean.repeat(b_dim, 1)
                z_std_t = self.z0_std.repeat(b_dim, 1)
            z_mean.append(z_mean_t)
            z_std.append(z_std_t)
            if sample:
                z_t = self._sample_gauss(z_mean_t, z_std_t)
            else:
                z_t = z_mean_t
        z_mean, z_std = torch.stack(z_mean), torch.stack(z_std)
        return z_mean, z_std
        
    def forward(self, inputs, lengths, sample=True):
        """Takes in (optionally missing) inputs and reconstructs them.

        inputs : dict of str : torch.tensor
           keys are modality names, tensors are (T, B, D)
           for max sequence length T, batch size B and input dims D
        lengths : list of int
           lengths of all input sequences in the batch
        sample: bool
           whether to sample from z_t (default) or return MAP estimate
        """
        b_dim, t_max = len(lengths), max(lengths)

        # Infer z_t from x_t without temporal information
        z_obs_mean, z_obs_std, z_obs_mask = self.encode(inputs)
        
        # Forward pass to sample from p(z_t) for all timesteps
        z_fwd_mean, z_fwd_std = self.generate(t_max, b_dim, sample)

        # Backward pass to sample p(z_t|x_t, ..., x_T)
        z_bwd_mean, z_bwd_std = [], []
        for t in reversed(range(t_max)):
            # Add p(z_t) to the PoE calculation
            z_mean_t = [z_fwd_mean[t]]
            z_std_t = [z_fwd_std[t]]
            masks = [torch.ones((b_dim,), dtype=torch.uint8).to(self.device)]

            # Add p(z_t|z_{t+1}) to the PoE calculation
            if len(z_bwd_mean) > 0:
                z_mean_t.append(z_bwd_mean[-1])
                z_std_t.append(z_bwd_mean[-1])
                masks.append(torch.ones((b_dim,), dtype=torch.uint8,
                                        device=self.device))
                        
            # Add p(z_t|x_t) to the PoE calculation
            z_mean_t.append(z_obs_mean[t])
            z_std_t.append(z_obs_std[t])
            masks.append(z_obs_mask[t])
                
            # Combine the Gaussian parameters using PoE
            z_mean_t = torch.stack(z_mean_t, dim=0)
            z_std_t = torch.stack(z_std_t, dim=0)
            mask = torch.stack(masks, dim=0)
            z_mean_t, z_var_t = \
                self.product_of_experts(z_mean_t, z_std_t.pow(2), mask)
            z_std_t = z_var_t.pow(0.5)

            # Sample params for p(z_{t-1}|z_t) under p(z_t|x_t, ..., x_T)
            z_bwd_mean_t, z_bwd_std_t = [], []
            for k in range(self.n_bwd_particles):
                if self.n_bwd_particles == 1 and not sample:
                    z_t = z_mean_t
                else:
                    z_t = self._sample_gauss(z_mean_t, z_std_t)
                z_bwd_mean_t_k, z_bwd_std_t_k = self.bwd(z_t)
                z_bwd_mean_t.append(z_bwd_mean_t_k)
                z_bwd_std_t.append(z_bwd_std_t_k)

            # Take average of sampled distributions
            z_bwd_mean_t = torch.stack(z_bwd_mean_t, dim=0)
            z_bwd_std_t = torch.stack(z_bwd_std_t, dim=0)
            z_bwd_mean_t, z_bwd_var_t = \
                self.mean_of_experts(z_bwd_mean_t, z_bwd_std_t.pow(2))
            z_bwd_std_t = z_bwd_var_t.pow(0.5)
            z_bwd_mean.append(z_bwd_mean_t)
            z_bwd_std.append(z_bwd_std_t)
            
        # Reverse lists that were accumulated backwards
        z_bwd_mean.reverse()
        z_bwd_std.reverse()
            
        # Final forward pass to infer p(z_1:T|x_1:T)
        prior_mean, prior_std = [], []
        infer_mean, infer_std = [], []
        z_final = []
        for t in range(t_max):
            # Compute params for p(z_t|z_{t-1})
            if t > 0:
                prior_mean_t, prior_std_t = self.fwd(z_t)
            else:
                prior_mean_t = self.z0_mean.repeat(b_dim, 1)
                prior_std_t = self.z0_std.repeat(b_dim, 1)
            prior_mean.append(prior_mean_t)
            prior_std.append(prior_std_t)

            # Concatenate p(z_t|z_{t-1}), p(z_t|x_t), p(z_t|z_{t+1})
            prior_mask = torch.ones((b_dim,), dtype=torch.uint8,
                                    device=self.device)
            z_mean_t = [z_obs_mean[t], prior_mean_t]
            z_std_t = [z_obs_std[t], prior_std_t]
            masks = [z_obs_mask[t], prior_mask]
            if t < t_max - 1:
                z_mean_t.append(z_bwd_mean[t])
                z_std_t.append(z_bwd_std[t])
                masks.append(torch.ones((b_dim,), dtype=torch.uint8,
                                        device=self.device))

            # Compute p(z_t|z_{t-1}, x_t, ..., x_T) using PoE
            z_mean_t = torch.stack(z_mean_t, dim=0)
            z_std_t = torch.stack(z_std_t, dim=0)
            mask = torch.stack(masks, dim=0)
            infer_mean_t, infer_var_t = \
                self.product_of_experts(z_mean_t, z_std_t.pow(2), mask)
            infer_std_t = infer_var_t.pow(0.5)
            infer_mean.append(infer_mean_t)
            infer_std.append(infer_std_t)

            if sample:
                # Sample z from p(z_t|z_{t-1}, x_t, ..., x_T)
                z_t = self._sample_gauss(infer_mean_t, infer_std_t)
            else:
                z_t = infer_mean_t
            z_final.append(z_t)

        # Decode sampled z to reconstruct inputs
        z_final = torch.stack(z_final)
        out_mean, out_std = self.decode(z_final)

        # Concatenate lists to tensors
        infer = (torch.stack(infer_mean), torch.stack(infer_std))
        prior = (torch.stack(prior_mean), torch.stack(prior_std))
        outputs = (out_mean, out_std)

        return infer, prior, outputs

    def sample(self, t_max, b_dim):
        """Generates a sequence of the input data by sampling."""
        z_mean, z_std = self.generate(t_max, b_dim, sample=True)
        out_mean, out_std = self.decode(z_mean)
        return out_mean, out_std

if __name__ == "__main__":
    # Test code by running 'python -m models.dmm' from base directory
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
    model = MultiDMM(['spiral-x', 'spiral-y'], [1, 1],
                     device=torch.device('cpu'))
    model.eval()
    print("Passing a sample through the model...")
    data, mask, lengths = seq_collate_dict([dataset[0]])
    infer, prior, outputs = model(data, lengths)
    out_mean, out_std = outputs
    print("Predicted:")
    for x, y in zip(out_mean['spiral-x'], out_mean['spiral-y']):
        print("{:+0.3f}, {:+0.3f}".format(x.item(), y.item()))
