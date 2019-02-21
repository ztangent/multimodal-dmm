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

class MultiVRNN(nn.Module):
    def __init__(self, modalities, dims, h_dim=256, z_dim=256,
                 n_layers=1, bias=False, device=torch.device('cuda:0')):
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

        # Feature-extracting transformations
        self.phi = nn.ModuleDict()
        for m in self.modalities:
            self.phi[m] = nn.Sequential(
                nn.Linear(self.dims[m], h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, h_dim),
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
                nn.ReLU(),
                nn.Linear(h_dim, h_dim),
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
                nn.ReLU(),
                nn.Linear(h_dim, h_dim),
                nn.ReLU())
            self.dec_mean[m] = nn.Linear(h_dim, self.dims[m])
            self.dec_std[m] = nn.Sequential(
                nn.Linear(h_dim, self.dims[m]),
                nn.Softplus())
        
        # Recurrence h_next = f(x,z,h)
        self.rnn = nn.GRU((self.n_mods + 1) * h_dim, h_dim, n_layers, bias)

        # Store module in specified device (CUDA/CPU)
        self.device = (device if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.to(self.device)

    def product_of_experts(self, mean, var, mask=None, eps=1e-8):
        """
        Return parameters for product of independent Gaussian experts.
        See https://arxiv.org/pdf/1410.7827.pdf for equations.

        mean : torch.tensor
            (M, B, D) for M experts, batch size B, and D latent dims
        var : torch.tensor
            (M, B, D) for M experts, batch size B, and D latent dims
        mask : torch.tensor
            (M, B) for M experts and batch size B
        """
        var = var + eps # numerical constant for stability
        # Precision matrix of i-th Gaussian expert (T = 1/sigma^2)
        T = 1. / var
        # Set missing data to zero so they are excluded from calculation
        if mask is None:
            mask = 1 - torch.isnan(var[:,:,0])
        T = T * mask.float().unsqueeze(-1)
        mean = mean * mask.float().unsqueeze(-1)
        product_mean = torch.sum(mean * T, dim=0) / torch.sum(T, dim=0)
        product_var = 1. / torch.sum(T, dim=0)
        return product_mean, product_var
        
    def forward(self, inputs, lengths):
        """Takes in (optionally missing) inputs and reconstructs them.

        inputs : dict of str : torch.tensor
           keys are modality names, tensors are (T, B, D)
           for max sequence length T, batch size B and input dims D
        lengths : list of int
           lengths of all input sequences in the batch
        """
        batch_size, seq_len = len(lengths), max(lengths)

        # Initialize list accumulators
        prior_mean, prior_std = [], []
        infer_mean, infer_std = [], []
        out_mean = {m: [] for m in self.modalities}
        out_std = {m: [] for m in self.modalities}
        
        # Initialize hidden state
        h = torch.zeros(self.n_layers, batch_size, self.h_dim).to(self.device)
            
        for t in range(seq_len):
            # Compute prior for z
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)
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

            # Sample z from approximate posterior q(z|x)
            zq_t = self._sample_gauss(infer_mean_t, infer_std_t)
            phi_zq_t = self.phi_z(zq_t)
            
            # Decode sampled z to reconstruct inputs
            dec_in_t = torch.cat([phi_zq_t, h[-1]], 1)
            for m in self.modalities:
                out_m_t = self.dec[m](dec_in_t)
                out_mean_m_t = self.dec_mean[m](out_m_t)
                out_std_m_t = self.dec_std[m](out_m_t)
                out_mean[m].append(out_mean_m_t)
                out_std[m].append(out_std_m_t)

            # Fill missing inputs with reconstructions then extract features
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
                
            # Compute h using filled-in x and inferred z (h_next = f(x,z,h))
            rnn_in = torch.cat([phi_x_t, phi_zq_t], 1)
            _, h = self.rnn(rnn_in.unsqueeze(0), h)

        # Concatenate lists to tensors
        infer = (torch.stack(infer_mean), torch.stack(infer_std))
        prior = (torch.stack(prior_mean), torch.stack(prior_std))
        for m in self.modalities:
            out_mean[m] = torch.stack(out_mean[m])
            out_std[m] = torch.stack(out_std[m])
        outputs = (out_mean, out_std)

        return infer, prior, outputs

    def sample(self, seq_len):
        """Generates a sequence of the input data by sampling."""
        out_mean = {m: [] for m in self.modalities}
        h = torch.zeros(self.n_layers, 1, self.h_dim).to(self.device)

        for t in range(seq_len):
            # Compute prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # Sample from prior
            z_t = self._sample_gauss(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)
            
            # Decode sampled z to reconstruct inputs
            dec_in_t = torch.cat([phi_z_t, h[-1]], 1)
            for m in self.modalities:
                out_m_t = self.dec[m](dec_in_t)
                out_mean_m_t = self.dec_mean[m](out_m_t)
                out_mean[m].append(out_mean_m_t)

            # Extract features from reconstructions
            phi_x_t = []
            for m in self.modalities:
                phi_m_t = self.phi[m](out_mean[m][-1])
                phi_x_t.append(phi_m_t)
            phi_x_t = torch.cat(phi_x_t, 1)
                
            # Recurrence h_next = f(x,z,h)
            rnn_in = torch.cat([phi_x_t, phi_z_t], 1)
            _, h = self.rnn(rnn_in.unsqueeze(0), h)

        for m in self.modalities:
            out_mean[m] = torch.stack(out_mean[m])
            
        return out_mean

    def loss(self, inputs, infer, prior, outputs, mask=1,
             kld_mult=1.0, rec_mults={}, avg=False):
        loss = 0.0
        loss += kld_mult * self.kld_loss(infer, prior, mask)
        loss += self.rec_loss(inputs, outputs, mask, rec_mults)
        if avg:
            if type(mask) is torch.Tensor:
                n_data = torch.sum(mask)
            else:
                n_data = inputs[self.modalities[-1]].numel()
            loss /= n_data
        return loss
    
    def kld_loss(self, infer, prior, mask=None):
        """KLD loss between inferred and prior z."""
        infer_mean, infer_std = infer
        prior_mean, prior_std = prior
        return self._kld_gauss(infer_mean, infer_std,
                               prior_mean, prior_std, mask)

    def rec_loss(self, inputs, outputs, mask=None, rec_mults={}):
        """Input reconstruction loss."""
        loss = 0.0
        out_mean, out_std = outputs
        for m in self.modalities:
            if m not in inputs:
                continue
            mult = 1.0 if m not in rec_mults else rec_mults[m]
            loss += mult * self._nll_gauss(out_mean[m], out_std[m],
                                           inputs[m], mask)
        return loss
            
    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)

    def _init_weights(self, stdv):
        pass

    def _sample_gauss(self, mean, std):
        """Use std to sample."""
        eps = torch.FloatTensor(std.size()).to(self.device).normal_()
        return eps.mul(std).add_(mean)

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2, mask=None):
        """Use std to compute KLD"""
        kld_element =  (2 * torch.log(std_2) - 2 * torch.log(std_1) + 
            (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
            std_2.pow(2) - 1)
        if mask is not None:
            kld_element = kld_element.masked_select(mask)
        kld =  0.5 * torch.sum(kld_element)
        return kld

    def _nll_bernoulli(self, theta, x, mask=None):
        nll_element = x*torch.log(theta) + (1-x)*torch.log(1-theta)
        if mask is None:
            mask = 1 - torch.isnan(x)
        else:
            mask = mask * (1 - torch.isnan(x))
        nll_element = nll_element.masked_select(mask)
        return torch.sum(nll_element)

    def _nll_gauss(self, mean, std, x, mask=None):
        if mask is None:
            mask = 1 - torch.isnan(x)
        else:
            mask = mask * (1 - torch.isnan(x))
        x = torch.tensor(x)
        x[torch.isnan(x)] = 0.0
        nll_element = ( ((x-mean).pow(2)) / (2 * std.pow(2)) + std.log() +
                        math.log(math.sqrt(2 * math.pi)) )
        nll_element = nll_element.masked_select(mask)
        nll = torch.sum(nll_element)
        return(nll)
    
if __name__ == "__main__":
    # Test code by loading dataset and running through model
    import os, argparse
    from datasets import load_spirals, seq_collate_dict

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default="../../data",
                        help='data directory')
    parser.add_argument('--subset', type=str, default="train",
                        help='whether to load train/test data')
    args = parser.parse_args()

    print("Loading data...")
    dataset = load_spirals(['spiral-x', 'spiral-y'],
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
    print("Predicted ratings:")
    for o, o_std in zip(out_mean['ratings'], out_std['ratings']):
        print("{:+0.3f} +- {:0.3f}".format(o.item(), o_std.item()))
