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

from .dgts import MultiDGTS

class MultiDMM(MultiDGTS):
    def __init__(self, modalities, dims, phi_dim=32, z_dim=32,
                 z0_mean=0.0, z0_std=1.0, n_bwd_particles=1,
                 device=torch.device('cuda:0')):
        """
        Construct multimodal deep Markov model.

        modalities : list of str
            list of names for each modality
        dims : list of int
            list of feature dimensions for each modality
        phi_dim : int
            size of intermediary layers
        z_dim : int
            number of latent dimensions
        device : torch.device
            device on which this module is stored (CPU or GPU)
        """
        super(MultiDMM, self).__init__()
        self.modalities = modalities
        self.n_mods = len(modalities)
        self.dims = dict(zip(modalities, dims))
        self.phi_dim = phi_dim
        self.z_dim = z_dim

        # Feature-extracting transformations
        self.phi = nn.ModuleDict()
        for m in self.modalities:
            self.phi[m] = nn.Sequential(
                nn.Linear(self.dims[m], phi_dim),
                nn.ReLU())
            
        # Encoders for each modality p(z|x) = N(mu(x), sigma(x))
        self.enc = nn.ModuleDict()
        self.enc_mean = nn.ModuleDict()
        self.enc_std = nn.ModuleDict()
        for m in self.modalities:
            self.enc[m] = nn.Sequential(
                nn.Linear(phi_dim, phi_dim),
                nn.ReLU())
            self.enc_mean[m] = nn.Linear(phi_dim, z_dim)
            self.enc_std[m] = nn.Sequential(
                nn.Linear(phi_dim, z_dim),
                nn.Softplus())

        # Decoders for each modality p(xi|z) = N(mu(z), sigma(z))
        self.dec = nn.ModuleDict()
        self.dec_mean = nn.ModuleDict()
        self.dec_std = nn.ModuleDict()
        for m in self.modalities:
            self.dec[m] = nn.Sequential(
                nn.Linear(z_dim, phi_dim),
                nn.ReLU())
            self.dec_mean[m] = nn.Linear(phi_dim, self.dims[m])
            self.dec_std[m] = nn.Sequential(
                nn.Linear(phi_dim, self.dims[m]),
                nn.Softplus())
            
        # Forward conditional p(z|z_prev) = N(mu(z_prev), sigma(z_prev))
        self.fwd = nn.Sequential(
            nn.Linear(z_dim, phi_dim),
            nn.ReLU())
        self.fwd_mean = nn.Linear(phi_dim, z_dim)
        self.fwd_std = nn.Sequential(
            nn.Linear(phi_dim, z_dim),
            nn.Softplus())

        # Backward conditional q(z|z_next) = N(mu(z_next), sigma(z_next))
        self.bwd = nn.Sequential(
            nn.Linear(z_dim, phi_dim),
            nn.ReLU())
        self.bwd_mean = nn.Linear(phi_dim, z_dim)
        self.bwd_std = nn.Sequential(
            nn.Linear(phi_dim, z_dim),
            nn.Softplus())

        # Number of sampling particles in backward pass
        self.n_bwd_particles = n_bwd_particles
        
        # Store module in specified device (CUDA/CPU)
        self.device = (device if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.to(self.device)

        # Initial prior
        self.z0_mean = z0_mean * torch.ones(1, z_dim).to(self.device)
        self.z0_std = z0_std * torch.ones(1, z_dim).to(self.device)        
    
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
        batch_size, seq_len = len(lengths), max(lengths)

        # Initialize list accumulators
        z_fwd_mean, z_fwd_std = [], []
        z_bwd_mean, z_bwd_std = [], []
        z_obs_mean, z_obs_std, z_obs_masks = [], [], []
        prior_mean, prior_std = [], []
        infer_mean, infer_std = [], []
        out_mean = {m: [] for m in self.modalities}
        out_std = {m: [] for m in self.modalities}
        
        # Forward pass to sample from p(z_t) for all timesteps
        for t in range(seq_len):
            # Compute params for p(z_t|z_{t-1})
            if t > 0:
                fwd_t = self.fwd(z_t)
                z_fwd_mean_t = self.fwd_mean(fwd_t)
                z_fwd_std_t = self.fwd_std(fwd_t)
            else:
                z_fwd_mean_t = self.z0_mean.repeat(batch_size, 1)
                z_fwd_std_t = self.z0_std.repeat(batch_size, 1)
            z_fwd_mean.append(z_fwd_mean_t)
            z_fwd_std.append(z_fwd_std_t)

            if sample:
                # Sample z_t from p(z_t|z_{t-1})
                z_t = self._sample_gauss(z_fwd_mean_t, z_fwd_std_t)
            else:
                z_t = z_fwd_mean_t

        # Backward pass to sample p(z_t|x_t, ..., x_T)
        for t in reversed(range(seq_len)):
            # Add p(z_t) to the PoE calculation
            z_mean_t = [z_fwd_mean[t]]
            z_std_t = [z_fwd_std[t]]
            masks = [torch.ones((batch_size,), dtype=torch.uint8,
                                device=self.device)]

            # Add p(z_t|z_{t+1}) to the PoE calculation
            if len(z_bwd_mean) > 0:
                z_mean_t.append(z_bwd_mean[-1])
                z_std_t.append(z_bwd_mean[-1])
                masks.append(torch.ones((batch_size,), dtype=torch.uint8,
                                        device=self.device))
                        
            # Encode modalities of x_t to latent code z_t
            z_obs_mean.append([])
            z_obs_std.append([])
            z_obs_masks.append([])
            for m in self.modalities:
                # Ignore missing modalities
                if m not in inputs:
                    continue
                # Mask out NaNs
                mask = (1 - torch.isnan(inputs[m][t]).any(dim=1))
                input_m_t = torch.tensor(inputs[m][t])
                input_m_t[torch.isnan(input_m_t)] = 0.0
                # Extract features 
                phi_m_t = self.phi[m](input_m_t)
                # Compute mean and std of latent z given modality m
                enc_m_t = self.enc[m](phi_m_t)
                z_mean_m_t = self.enc_mean[m](enc_m_t)
                z_std_m_t = self.enc_std[m](enc_m_t)
                z_obs_mean[-1].append(z_mean_m_t)
                z_obs_std[-1].append(z_std_m_t)
                z_obs_masks[-1].append(mask)
                # Add p(z_t|x_{t,m}) to the PoE calculation
                z_mean_t.append(z_mean_m_t)
                z_std_t.append(z_std_m_t)
                masks.append(mask)
                
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
                bwd_t = self.bwd(z_t)
                z_bwd_mean_t.append(self.bwd_mean(bwd_t))
                z_bwd_std_t.append(self.bwd_std(bwd_t))

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
        z_obs_mean.reverse()
        z_obs_std.reverse()
        z_obs_masks.reverse()
            
        # Final forward pass to infer p(z_1:T|x_1:T) and reconstruct x_1:T
        for t in range(seq_len):
            # Compute params for p(z_t|z_{t-1})
            if t > 0:
                fwd_t = self.fwd(z_t)
                prior_mean_t = self.fwd_mean(fwd_t)
                prior_std_t = self.fwd_std(fwd_t)
            else:
                prior_mean_t = self.z0_mean.repeat(batch_size, 1)
                prior_std_t = self.z0_std.repeat(batch_size, 1)
            prior_mean.append(prior_mean_t)
            prior_std.append(prior_std_t)

            # Concatenate p(z_t|z_{t-1}), p(z_t|x_t), p(z_t|z_{t+1})
            z_mean_t = list(z_obs_mean[t]) + [prior_mean_t]
            z_std_t = list(z_obs_std[t]) + [prior_std_t]
            prior_mask = torch.ones((batch_size,), dtype=torch.uint8,
                                    device=self.device)
            masks = list(z_obs_masks[t]) + [prior_mask]
            if t < seq_len - 1:
                z_mean_t.append(z_bwd_mean[t])
                z_std_t.append(z_bwd_std[t])
                masks.append(torch.ones((batch_size,), dtype=torch.uint8,
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

            # Decode sampled z to reconstruct inputs
            for m in self.modalities:
                out_m_t = self.dec[m](z_t)
                out_mean_m_t = self.dec_mean[m](out_m_t)
                out_std_m_t = self.dec_std[m](out_m_t)
                out_mean[m].append(out_mean_m_t)
                out_std[m].append(out_std_m_t)

        # Concatenate lists to tensors
        infer = (torch.stack(infer_mean), torch.stack(infer_std))
        prior = (torch.stack(prior_mean), torch.stack(prior_std))
        for m in self.modalities:
            out_mean[m] = torch.stack(out_mean[m])
            out_std[m] = torch.stack(out_std[m])
        outputs = (out_mean, out_std)

        return infer, prior, outputs

    def sample(self, batch_size, seq_len):
        """Generates a sequence of the input data by sampling."""
        out_mean = {m: [] for m in self.modalities}
        z_t = self.z0.repeat(batch_size, 1)

        for t in range(seq_len):
            # Compute prior
            if t > 0:
                fwd_t = self.fwd(z_t)
                prior_mean_t = self.fwd_mean(fwd_t)
                prior_std_t = self.fwd_std(fwd_t)
            else:
                prior_mean_t = self.z0_mean.repeat(batch_size, 1)
                prior_std_t = self.z0_std.repeat(batch_size, 1)

            # Sample from prior
            z_t = self._sample_gauss(prior_mean_t, prior_std_t)
            
            # Decode sampled z to reconstruct inputs
            for m in self.modalities:
                out_m_t = self.dec[m](z_t)
                out_mean_m_t = self.dec_mean[m](out_m_t)
                out_mean[m].append(out_mean_m_t)

        for m in self.modalities:
            out_mean[m] = torch.stack(out_mean[m])
            
        return out_mean

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
