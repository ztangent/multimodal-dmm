"""Multimodal Deep Kalman Smoother (DKS).

Original DKS described by Krishan et. al. (https://arxiv.org/abs/1609.09869)

We extend the DKS to multiple modalities by having an RNN inference network
for each modality. When observations for a particular modality are unobserved,
we do not update the hidden state.

Requires pytorch >= 0.4.1 for nn.ModuleDict
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import math
import torch
import torch.nn as nn

from .dgts import MultiDGTS

class MultiDKS(MultiDGTS):
    def __init__(self, modalities, dims, dists=None, h_dim=32, z_dim=32,
                 z0_mean=0.0, z0_std=1.0, n_layers=1, bias=True,
                 device=torch.device('cuda:0')):
        """
        Construct multimodal deep Markov model.

        modalities : list of str
            list of names for each modality
        dims : list of int
            list of feature dimensions for each modality
        dists : list of str
            list of distributions ('Normal' [default] or 'Bernoulli')
        h_dim : int
            size of intermediary layers and RNN hidden state
        z_dim : int
            number of latent dimensions
        n_layers : int
            number of RNN layers
        bias : bool
            whether RNN should learn a bias
        device : torch.device
            device on which this module is stored (CPU or GPU)
        """
        super(MultiDKS, self).__init__()
        self.modalities = modalities
        self.n_mods = len(modalities)
        self.dims = dict(zip(modalities, dims))
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers

        # Default to Gaussian distributions
        if dists is None:
            dists = ['Normal'] * self.n_mods
        self.dists = dict(zip(modalities, dists))

        # Feature-extracting transformations
        self.phi = nn.ModuleDict()
        for m in self.modalities:
            self.phi[m] = nn.Sequential(
                nn.Linear(self.dims[m], h_dim),
                nn.ReLU())
        
        # Decoders for each modality p(xi|z) = N(mu(z), sigma(z))
        self.dec = nn.ModuleDict()
        self.dec_mean = nn.ModuleDict()
        self.dec_std = nn.ModuleDict()
        for m in self.modalities:
            self.dec[m] = nn.Sequential(
                nn.Linear(z_dim, h_dim),
                nn.ReLU())
            self.dec_mean[m] = nn.Linear(h_dim, self.dims[m])
            self.dec_std[m] = nn.Sequential(
                nn.Linear(h_dim, self.dims[m]),
                nn.Softplus())
            
        # Forward conditional p(z|z_prev) = N(mu(z_prev), sigma(z_prev))
        self.fwd = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU())
        self.fwd_mean = nn.Linear(h_dim, z_dim)
        self.fwd_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        # Backwards inference RNNs h_prev = f(x,h)
        self.rnn = nn.ModuleDict()
        self.h0 = nn.ParameterDict()
        for m in self.modalities:
            self.rnn[m] = nn.GRU(h_dim, h_dim, n_layers, bias)
            self.h0[m] = nn.Parameter(torch.zeros(n_layers, 1, h_dim))

        # Combiner inference network q(z) = N(mu(z_prev, h), sigma(z_prev, h))
        self.z_to_comb = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.Tanh())
        self.comb_mean = nn.Linear(h_dim, z_dim)
        self.comb_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        # Store module in specified device (CUDA/CPU)
        self.device = (device if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.to(self.device)

        # Initial prior
        self.z0_mean = z0_mean * torch.ones(1, z_dim).to(self.device)
        self.z0_std = z0_std * torch.ones(1, z_dim).to(self.device)        
    
    def forward(self, inputs, **kwargs):
        """Takes in (optionally missing) inputs and reconstructs them.

        inputs : dict of str : torch.tensor
           keys are modality names, tensors are (T, B, D)
           for max sequence length T, batch size B and input dims D
        lengths : list of int
           lengths of all input sequences in the batch
        sample: bool
           whether to sample from z_t (default) or return MAP estimate
        """
        lengths, sample = kwargs.get('lengths'), kwargs.get('sample', True)
        batch_size, seq_len = len(lengths), max(lengths)

        # Initialize list accumulators
        prior_mean, prior_std = [], []
        infer_mean, infer_std = [], []
        out_mean = {m: [] for m in self.modalities}
        out_std = {m: [] for m in self.modalities}

        # Zero mask missing values and extract features
        feats, masks = dict(), dict()
        for m in self.modalities:
            if m not in inputs:
                input_m = torch.zeros(seq_len, batch_size,
                                      self.dims[m]).to(self.device)
                masks[m] = torch.zeros(seq_len, batch_size).to(self.device)
            else:
                input_m = torch.tensor(inputs[m])
                masks[m] = (1 - torch.isnan(inputs[m])).any(dim=2).float()
            input_m[torch.isnan(input_m)] = 0.0
            input_m = input_m.view(-1, self.dims[m])
            feats[m] = self.phi[m](input_m).reshape(seq_len, batch_size, -1)

        # Initialize RNN hidden states
        h = {m: self.h0[m].repeat(1, batch_size, 1) for m in self.modalities}
        h_out = {m: [] for m in self.modalities}
        
        # Backward pass through RNN inference networks
        for t in reversed(range(seq_len)):
            for m in self.modalities:
                _, h_m_next = self.rnn[m](feats[m][t:t+1], h[m])
                # Only update if modality m is observed at time t
                mask_m = masks[m][t].view(1, batch_size, 1)
                h[m] = mask_m * h_m_next + (1-mask_m) * h[m]
                h_out[m].append(h[m][-1])

        # Concatenate RNN outputs, sum across modalities, then flip
        h_out = torch.stack([torch.stack(h_out[m], dim=0)
                             for m in self.modalities], dim=-1).mean(dim=-1)
        h_out = torch.flip(h_out, [0])
                
        # Forward pass to infer p(z_1:T|x_1:T) and reconstruct x_1:T
        for t in range(seq_len):
            # Compute params for the prior p(z_t|z_{t-1})
            if t > 0:
                fwd_t = self.fwd(z_t)
                prior_mean_t = self.fwd_mean(fwd_t)
                prior_std_t = self.fwd_std(fwd_t)
            else:
                prior_mean_t = self.z0_mean.repeat(batch_size, 1)
                prior_std_t = self.z0_std.repeat(batch_size, 1)
                z_t = prior_mean_t
            prior_mean.append(prior_mean_t)
            prior_std.append(prior_std_t)

            # Infer the latent distribution p(z_t|z_{t-1}, x_t, ..., x_T)
            h_comb = 0.5 * (self.z_to_comb(z_t) + h_out[t])
            infer_mean_t = self.comb_mean(h_comb)
            infer_std_t = self.comb_std(h_comb)
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
    # Test code by running 'python -m models.dks' from base directory
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
    model = MultiDKS(['spiral-x', 'spiral-y'], [1, 1],
                     device=torch.device('cpu'))
    model.eval()
    print("Passing a sample through the model...")
    data, mask, lengths, order = seq_collate_dict([dataset[0]])
    infer, prior, outputs = model(data, lengths=lengths)
    out_mean, out_std = outputs
    print("Predicted:")
    for x, y in zip(out_mean['spiral-x'], out_mean['spiral-y']):
        print("{:+0.3f}, {:+0.3f}".format(x.item(), y.item()))
