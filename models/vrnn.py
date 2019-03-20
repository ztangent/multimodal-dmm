"""Multimodal Variational Recurrent Neural Network, adapted from
https://github.com/emited/VariationalRecurrentNeuralNetwork

Original VRNN described in https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models.

To handle missing modalities, we use the MVAE approach
described in https://arxiv.org/abs/1802.05335.

Requires pytorch >= 0.4.1 for nn.ModuleDict
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import math
import torch
import torch.nn as nn

from .dgts import MultiDGTS

class MultiVRNN(MultiDGTS):
    def __init__(self, modalities, dims, h_dim=16, z_dim=16,
                 z0_mean=0.0, z0_std=1.0, n_layers=1, bias=True,
                 recur_mode='no_inputs', device=torch.device('cuda:0')):
        """
        Construct multimodal variational recurrent neural network.

        modalities : list of str
            list of names for each modality
        dims : list of int
            list of feature dimensions for each modality
        h_dim : int
            number of hidden dimensions
        z_dim : int
            number of latent dimensions
        n_layers : int
            number of RNN layers
        bias : bool
            whether RNN should learn a bias
        recur_mode : str ('use_inputs', 'no_inputs')
            whether h should be a function of x
        device : torch.device
            device on which this module is stored (CPU or GPU)
        """
        super(MultiVRNN, self).__init__()
        self.modalities = modalities
        self.n_mods = len(modalities)
        self.dims = dict(zip(modalities, dims))
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.recur_mode = recur_mode

        # Feature-extracting transformations
        self.phi = nn.ModuleDict()
        for m in self.modalities:
            self.phi[m] = nn.Sequential(
                nn.Linear(self.dims[m], h_dim),
                nn.ReLU())
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU())

        # Encoder p(z|x) = N(mu(x,h), sigma(x,h))
        self.enc = nn.ModuleDict()
        self.enc_mean = nn.ModuleDict()
        self.enc_std = nn.ModuleDict()
        for m in self.modalities:
            self.enc[m] = nn.Sequential(
                nn.Linear(h_dim + h_dim, h_dim),
                nn.ReLU())
            self.enc_mean[m] = nn.Linear(h_dim, z_dim)
            self.enc_std[m] = nn.Sequential(
                nn.Linear(h_dim, z_dim),
                nn.Softplus())

        # Prior p(z) = N(mu(h), sigma(h))
        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.prior_mean = nn.Linear(h_dim, z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        # Decoders p(xi|z) = N(mu(z,h), sigma(z,h))
        self.dec = nn.ModuleDict()
        self.dec_mean = nn.ModuleDict()
        self.dec_std = nn.ModuleDict()
        for m in self.modalities:
            self.dec[m] = nn.Sequential(
                nn.Linear(h_dim + h_dim, h_dim),
                nn.ReLU())
            self.dec_mean[m] = nn.Linear(h_dim, self.dims[m])
            self.dec_std[m] = nn.Sequential(
                nn.Linear(h_dim, self.dims[m]),
                nn.Softplus())
        
        # Recurrence h_next = f(z,h) or f(x,z,h)
        if recur_mode == 'use_inputs':
            self.rnn = nn.GRU((self.n_mods + 1) * h_dim, h_dim, n_layers, bias)
        else:
            self.rnn = nn.GRU(h_dim, h_dim, n_layers, bias)
        self.h0 = nn.Parameter(torch.zeros(self.n_layers, 1, self.h_dim))
        
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
        prior_mean, prior_std = [], []
        infer_mean, infer_std = [], []
        out_mean = {m: [] for m in self.modalities}
        out_std = {m: [] for m in self.modalities}
        
        # Initialize hidden state
        h = self.h0.repeat(1, batch_size, 1)
            
        for t in range(seq_len):
            # Compute prior for z
            if t > 0:
                prior_t = self.prior(h[-1])
                prior_mean_t = self.prior_mean(prior_t)
                prior_std_t = self.prior_std(prior_t)
            else:
                prior_mean_t = self.z0_mean.repeat(batch_size, 1)
                prior_std_t = self.z0_std.repeat(batch_size, 1)
            prior_mean.append(prior_mean_t)
            prior_std.append(prior_std_t)

            # Accumulate list of the means and std for z
            z_mean_t = [prior_mean_t]
            z_std_t = [prior_std_t]
            masks = [torch.ones((batch_size,), dtype=torch.uint8,
                                device=self.device)]
            
            # Encode modalities to latent code z
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
                enc_in_t = torch.cat([phi_m_t, h[-1]], 1)
                # Compute mean and std of latent z given modality m
                enc_m_t = self.enc[m](enc_in_t)
                z_mean_m_t = self.enc_mean[m](enc_m_t)
                z_std_m_t = self.enc_std[m](enc_m_t)
                # Concatenate to list of inferred means and stds
                z_mean_t.append(z_mean_m_t)
                z_std_t.append(z_std_m_t)
                masks.append(mask)

            # Combine the inferred distributions from each modality using PoE
            z_mean_t = torch.stack(z_mean_t, dim=0)
            z_std_t = torch.stack(z_std_t, dim=0)
            mask = torch.stack(masks, dim=0)
            infer_mean_t, infer_var_t = \
                self.product_of_experts(z_mean_t, z_std_t.pow(2), mask)
            infer_std_t = infer_var_t.pow(0.5)
            
            infer_mean.append(infer_mean_t)
            infer_std.append(infer_std_t)

            if sample:
                # Sample z from approximate posterior q(z|x)
                zq_t = self._sample_gauss(infer_mean_t, infer_std_t)
            else:
                zq_t = infer_mean_t
            phi_zq_t = self.phi_z(zq_t)
            
            # Decode sampled z to reconstruct inputs
            dec_in_t = torch.cat([phi_zq_t, h[-1]], 1)
            for m in self.modalities:
                out_m_t = self.dec[m](dec_in_t)
                out_mean_m_t = self.dec_mean[m](out_m_t)
                out_std_m_t = self.dec_std[m](out_m_t)
                out_mean[m].append(out_mean_m_t)
                out_std[m].append(out_std_m_t)

            if self.recur_mode == 'use_inputs':
                # Impute missing inputs then extract features
                phi_x_t = []
                for m in self.modalities:
                    if m not in inputs:
                        input_m_t = out_mean[m][-1].detach()
                    else:
                        input_m_t = torch.tensor(inputs[m][t])
                        nan_mask = torch.isnan(input_m_t)
                        input_m_t[nan_mask] = out_mean[m][-1][nan_mask]
                    phi_m_t = self.phi[m](input_m_t)
                    phi_x_t.append(phi_m_t)
                phi_x_t = torch.cat(phi_x_t, 1)

                # Compute h using imputed x and inferred z (h_next = f(x,z,h))
                rnn_in = torch.cat([phi_x_t, phi_zq_t], 1)
                _, h = self.rnn(rnn_in.unsqueeze(0), h)
                
            else:
                # Compute h using inferred z (h_next = f(x,z,h))
                _, h = self.rnn(phi_zq_t.unsqueeze(0), h)

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
        h = self.h0.repeat(1, batch_size, 1)

        for t in range(seq_len):
            # Compute prior
            if t > 0:
                prior_t = self.prior(h[-1])
                prior_mean_t = self.prior_mean(prior_t)
                prior_std_t = self.prior_std(prior_t)
            else:
                prior_mean_t = self.z0_mean.repeat(batch_size, 1)
                prior_std_t = self.z0_std.repeat(batch_size, 1)

            # Sample from prior
            z_t = self._sample_gauss(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)
            
            # Decode sampled z to reconstruct inputs
            dec_in_t = torch.cat([phi_z_t, h[-1]], 1)
            for m in self.modalities:
                out_m_t = self.dec[m](dec_in_t)
                out_mean_m_t = self.dec_mean[m](out_m_t)
                out_mean[m].append(out_mean_m_t)

            if self.recur_mode == 'use_inputs':
                # Extract features from reconstructions
                phi_x_t = []
                for m in self.modalities:
                    phi_m_t = self.phi[m](out_mean[m][-1])
                    phi_x_t.append(phi_m_t)
                phi_x_t = torch.cat(phi_x_t, 1)

                # Recurrence h_next = f(x,z,h)
                rnn_in = torch.cat([phi_x_t, phi_z_t], 1)
                _, h = self.rnn(rnn_in.unsqueeze(0), h)
            else:
                _, h = self.rnn(phi_z_t.unsqueeze(0), h)

        for m in self.modalities:
            out_mean[m] = torch.stack(out_mean[m])
            
        return out_mean
    
if __name__ == "__main__":
    # Test code by running 'python -m models.vrnn' from base directory
    import argparse
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
    model = MultiVRNN(['spiral-x', 'spiral-y'], [1, 1],
                      device=torch.device('cpu'))
    model.eval()
    print("Passing a sample through the model...")
    data, mask, lengths = seq_collate_dict([dataset[0]])
    infer, prior, outputs = model(data, lengths)
    out_mean, out_std = outputs
    print("Predicted:")
    for x, y in zip(out_mean['spiral-x'], out_mean['spiral-y']):
        print("{:+0.3f}, {:+0.3f}".format(x.item(), y.item()))
