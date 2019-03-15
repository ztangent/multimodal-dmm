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

class MultiDMM(nn.Module):
    def __init__(self, modalities, dims, phi_dim=32, z_dim=32,
                 n_layers=1, bias=False, device=torch.device('cuda:0')):
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
        n_layers : int
            number of RNN layers
        bias : bool
            whether RNN should learn a bias
        device : torch.device
            device on which this module is stored (CPU or GPU)
        """
        super(MultiDMM, self).__init__()
        self.modalities = modalities
        self.n_mods = len(modalities)
        self.dims = dict(zip(modalities, dims))
        self.phi_dim = phi_dim
        self.z_dim = z_dim
        self.n_layers = n_layers

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

        # Initial latent state
        self.z0 = nn.Parameter(torch.zeros(1, z_dim))
            
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
        z_fwd_mean, z_fwd_std = [], []
        z_bwd_mean, z_bwd_std = [], []
        z_obs_mean, z_obs_std, z_obs_masks = [], [], []
        prior_mean, prior_std = [], []
        infer_mean, infer_std = [], []
        out_mean = {m: [] for m in self.modalities}
        out_std = {m: [] for m in self.modalities}
        
        # Initialize latent state
        z_t = self.z0.repeat(batch_size, 1)
        # Forward pass to sample from p(z_t) for all timesteps
        for t in range(seq_len):
            # Compute params for p(z_t|z_{t-1})
            fwd_t = self.fwd(z_t)
            z_fwd_mean_t = self.fwd_mean(fwd_t)
            z_fwd_std_t = self.fwd_std(fwd_t)
            z_fwd_mean.append(z_fwd_mean_t)
            z_fwd_std.append(z_fwd_std_t)
            
            # Sample z_t from p(z_t|z_{t-1})
            z_t = self._sample_gauss(z_fwd_mean_t, z_fwd_std_t)

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
            
            # Sample z_t from p(z_t|x_t, ..., x_T)
            z_t = self._sample_gauss(z_mean_t, z_std_t)

            # Compute params for p(z_{t-1}|z_t)
            bwd_t = self.bwd(z_t)
            z_bwd_mean_t = self.bwd_mean(bwd_t)
            z_bwd_std_t = self.bwd_std(bwd_t)
            z_bwd_mean.append(z_bwd_mean_t)
            z_bwd_std.append(z_bwd_std_t)
            
        # Reverse lists that were accumulated backwards
        z_bwd_mean.reverse()
        z_bwd_std.reverse()
        z_obs_mean.reverse()
        z_obs_std.reverse()
        z_obs_masks.reverse()
            
        # Re-initialize latent state
        z_t = self.z0.repeat(batch_size, 1)
        # Final forward pass to infer p(z_1:T|x_1:T) and reconstruct x_1:T
        for t in range(seq_len):
            # Compute params for p(z_t|z_{t-1})
            prior_t = self.fwd(z_t)
            prior_mean_t = self.fwd_mean(prior_t)
            prior_std_t = self.fwd_std(prior_t)
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
            
            # Sample z from p(z_t|z_{t-1}, x_t, ..., x_T)
            z_t = self._sample_gauss(infer_mean_t, infer_std_t)

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
            prior_t = self.fwd(z_t)
            prior_mean_t = self.fwd_mean(prior_t)
            prior_std_t = self.fwd_std(prior_t)

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
    import os, sys, argparse
    parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    sys.path.insert(0, parent_dir)
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
