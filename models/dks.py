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

import numpy as np
import torch
import torch.nn as nn

from .dgts import MultiDGTS
from . import common
from datasets.multiseq import mask_to_extent

class MultiDKS(MultiDGTS):
    def __init__(self, modalities, dims, dists=None,
                 encoders=None, decoders=None, h_dim=32, z_dim=32,
                 z0_mean=0.0, z0_std=1.0, min_std=1e-3,
                 rnn_dir='bwd', rnn_skip=True, rnn_layers=1, rnn_bias=True,
                 device=torch.device('cuda:0')):
        """
        Construct multimodal deep Markov model.

        Parameters
        ----------         
        modalities : list of str
            list of names for each modality
        dims : list of int or tuple
            list of feature dimensions for each modality
        dists : list of str
            list of either 'Normal' [default], 'Bernoulli' or 'Categorical'
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
        rnn_dir : {'bwd', 'fwd'}
            whether inference RNN should run backwards or forwards
        rnn_skip : bool
            skip updates for missing data (True) or zero-mask (False)
        rnn_layers : int
            number of RNN layers
        rnn_bias : bool
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

        # Default to Gaussian distributions
        if dists is None:
            dists = ['Normal'] * self.n_mods
        self.dists = dict(zip(modalities, dists))

        # Encoders for each modality to RNN input features
        self.enc = nn.ModuleDict()            
        # Default to linear encoder with ReLU
        for m in self.modalities:
            if self.dists[m] == 'Categorical':
                self.enc[m] = nn.Sequential(
                    nn.Embedding(np.prod(self.dims[m]), h_dim),
                    nn.ReLU(),
                    nn.Linear(h_dim, h_dim),
                    nn.ReLU())
            else:
                self.enc[m] = nn.Sequential(
                    nn.Linear(np.prod(self.dims[m]), h_dim),
                    nn.ReLU())
        if encoders is not None:
            # Use custom encoders if provided
            if type(encoders) is list:
                encoders = zip(modalities, encoders)
            self.enc.update(encoders)
        
        # Decoders for each modality p(x|z) = N(mu(z), sigma(z))
        self.dec = nn.ModuleDict()
        # Default to MLP
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
                decoders = zip(modalities, decoders)
            self.dec.update(decoders)
            
        # Forward conditional p(z|z_prev) = N(mu(z_prev), sigma(z_prev))
        self.fwd = common.GaussianGTF(z_dim, h_dim, min_std=min_std)

        # Strutured inference RNNs h_prev = f(x,h)
        self.rnn_dir = rnn_dir
        self.rnn_skip = rnn_skip
        self.rnn = nn.ModuleDict()
        self.h0 = nn.ParameterDict()
        for m in self.modalities:
            if hasattr(self.enc[m], 'feat_dim'):
                feat_dim = self.enc[m].feat_dim
            else:
                feat_dim = h_dim
            self.rnn[m] = nn.GRU(feat_dim, h_dim, rnn_layers, rnn_bias)
            self.h0[m] = nn.Parameter(torch.zeros(rnn_layers, 1, h_dim))

        # Combiner inference network q(z) = N(mu(z_prev, h), sigma(z_prev, h))
        self.combiner = common.GaussianMLP(in_dim=z_dim + self.n_mods*h_dim,
                                           out_dim=z_dim, h_dim=h_dim)

        # Store module in specified device (CUDA/CPU)
        self.device = (device if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.to(self.device)

        # Initial prior
        self.z0_mean = z0_mean * torch.ones(1, z_dim).to(self.device)
        self.z0_std = z0_std * torch.ones(1, z_dim).to(self.device)        
    
    def forward(self, inputs, **kwargs):
        """Takes in (optionally missing) inputs and reconstructs them.

        Parameters
        ----------         
        inputs : dict of str : torch.tensor
           keys are modality names, tensors are (T, B, D)
           for max sequence length T, batch size B and input dims D
        lengths : list of int
           lengths of all input sequences in the batch
        sample: bool
           whether to sample from z_t (default) or return MAP estimate

        Returns
        -------
        infer : (torch.tensor, torch.tensor)
            (mean, std) of time-wise cond. posterior q(z_t|z_{t-1}, x)
        prior : (torch.tensor, torch.tensor)
            (mean, std) of time-wise cond. prior p(z_t|z_{t-1})
        recon : dict of str : (torch.tensor, ...)
           tuple of reconstructed distribution parameters for each modality
        """
        lengths, sample = kwargs.get('lengths'), kwargs.get('sample', True)
        b_dim, t_max = len(lengths), max(lengths)

        # Initialize list accumulators
        prior_mean, prior_std = [], []
        infer_mean, infer_std = [], []

        # Zero mask missing values and encode to features
        feats, masks = dict(), dict()
        for m in self.modalities:
            if m not in inputs:
                if self.dists[m] == 'Categorical':
                    input_m = torch.zeros(t_max, b_dim, 1)
                elif type(self.dims[m]) == tuple:
                    input_m = torch.zeros(t_max, b_dim, *self.dims[m])
                else:
                    input_m = torch.zeros(t_max, b_dim, self.dims[m])
                input_m = input_m.to(self.device)
                masks[m] = torch.zeros(t_max, b_dim).byte().to(self.device)
            else:
                input_m = inputs[m].clone().detach()
                masks[m] = 1 - torch.isnan(inputs[m]).flatten(2,-1).any(dim=-1)
            input_m[torch.isnan(input_m)] = 0.0
            if self.dists[m] == 'Categorical':
                input_m = input_m.long()
            # Flatten time and batch dimensions to pass through encoder
            input_m = input_m.flatten(0, 1)
            feats[m] = self.enc[m](input_m).reshape(t_max, b_dim, -1)
            
        # Initialize RNN hidden states
        h = {m: self.h0[m].repeat(1, b_dim, 1) for m in self.modalities}
        h_out = {m: [] for m in self.modalities}
        
        # Pass through RNN inference networks
        t_rng = range(t_max) if self.rnn_dir=='fwd' else reversed(range(t_max))
        for t in t_rng:
            for m in self.modalities:
                _, h_m_next = self.rnn[m](feats[m][t:t+1], h[m])
                if self.rnn_skip:
                    # Only update if modality m is observed at time t
                    mask_m = masks[m][t].view(1, b_dim, 1).float()
                    h[m] = mask_m * h_m_next + (1-mask_m) * h[m]
                else:
                    # Compute update with zero-masked inputs
                    h[m] = h_m_next
                h_out[m].append(h[m][-1])

        # Stack hidden states across time dimension
        h_out = {m: torch.stack(h_out[m], dim=0) for m in self.modalities}
        # Concatenate RNN outputs from each modality
        h_out = torch.cat([h_out[m] for m in self.modalities], dim=-1)
        # Flip across time if using backwards RNN
        if self.rnn_dir == 'bwd':
            h_out = torch.flip(h_out, [0])
                
        # Find indices for last observations
        mask_all = torch.stack([masks[m] for m in self.modalities]).prod(dim=0)
        _, t_stop = mask_to_extent(mask_all)
        t_stop = t_stop.unsqueeze(-1)
        
        # Forward pass to infer and sample from p(z_1:T|x_1:T)
        z_samples = []
        for t in range(t_max):
            # Compute params for the prior p(z_t|z_{t-1})
            if t > 0:
                prior_mean_t, prior_std_t = self.fwd(z_t)
            else:
                prior_mean_t = self.z0_mean.repeat(b_dim, 1)
                prior_std_t = self.z0_std.repeat(b_dim, 1)
                z_t = prior_mean_t
            prior_mean.append(prior_mean_t)
            prior_std.append(prior_std_t)

            # Infer the latent distribution p(z_t|z_{t-1}, x_{1:T})
            comb_in = torch.cat([z_t, h_out[t]], dim=-1)
            infer_mean_t, infer_std_t = self.combiner(comb_in)
            
            # Only infer for timesteps before the last observation
            infer_mean_t = (infer_mean_t * (t <= t_stop).float() + 
                            prior_mean_t * (t > t_stop).float())
            infer_std_t = (infer_std_t * (t <= t_stop).float() + 
                           prior_std_t * (t > t_stop).float())
            
            infer_mean.append(infer_mean_t)
            infer_std.append(infer_std_t)
            
            if sample:
                # Sample z from p(z_t|z_{t-1}, x_{1:T})
                z_t = self._sample_gauss(infer_mean_t, infer_std_t)
            else:
                z_t = infer_mean_t
            z_samples.append(z_t)

        # Concatenate z samples across time
        z_samples = torch.stack(z_samples, dim=0)

        # Decode sampled z to reconstruct (probability dist over) inputs
        recon = dict()
        for m in self.modalities:
            recon_m = self.dec[m](z_samples.view(-1, self.z_dim))
            rec_shape = [t_max, b_dim] + list(recon_m[0].shape[1:])
            # Reshape each output parameter (e.g. mean, std) to (T, B, ...)
            recon[m] = tuple(r.reshape(*rec_shape) for r in recon_m)

        # Concatenate lists to tensors
        infer = (torch.stack(infer_mean), torch.stack(infer_std))
        prior = (torch.stack(prior_mean), torch.stack(prior_std))

        return infer, prior, recon

    def sample(self, t_max, b_dim):
        """Generates a sequence of the input data by sampling.

        Parameters
        ----------         
        t_max : int
            number of timesteps T to sample
        b_dim : int
            batch size B

        Returns
        -------
        recon : dict of str : (torch.tensor, ...)
           tuple of reconstructed distribution parameters for each modality
        """
        rec_mean = {m: [] for m in self.modalities}

        # Forward pass to sample from p(z_1:T)
        z_samples = []
        for t in range(t_max):
            # Compute params for the prior p(z_t|z_{t-1})
            if t > 0:
                prior_mean_t, prior_std_t = self.fwd(z_t)
            else:
                prior_mean_t = self.z0_mean.repeat(b_dim, 1)
                prior_std_t = self.z0_std.repeat(b_dim, 1)
                z_t = prior_mean_t

            # Sample from prior
            z_t = self._sample_gauss(prior_mean_t, prior_std_t)
            z_samples.append(z_t)
            
        # Concatenate z samples across time
        z_samples = torch.stack(z_samples, dim=0)

        # Decode sampled z to reconstruct (probability dist over) inputs
        recon = dict()
        for m in self.modalities:
            recon_m = self.dec[m](z_samples.view(-1, self.z_dim))
            rec_shape = [t_max, b_dim] + list(recon_m[0].shape[1:])
            # Reshape each output parameter (e.g. mean, std) to (T, B, ...)
            recon[m] = tuple(r.reshape(*rec_shape) for r in recon_m)
            
        return recon

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
                             args.dir, args.subset,
                             truncate=True, item_as_dict=True)
    print("Building model...")
    model = MultiDKS(['spiral-x', 'spiral-y'], [1, 1],
                     device=torch.device('cpu'))
    model.eval()
    print("Passing a sample through the model...")
    data, mask, lengths, order = seq_collate_dict([dataset[0]])
    infer, prior, recon = model(data, lengths=lengths)
    print("Predicted:")
    for x, y in zip(recon['spiral-x'][0], recon['spiral-y'][0]):
        print("{:+0.3f}, {:+0.3f}".format(x.item(), y.item()))
