"""Training code for multimodal sequential data."""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys, os, shutil
import argparse, yaml

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from datasets import multiseq as mseq

import models
from utils import anneal, plot_grad_flow

class Trainer(object):
    """Abstract base class for training on multimodal sequential data."""

    def __init__(self, args):
        # Fix random seed
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)

        # Check for gradient anomalies
        if args.anomaly_check:
            torch.autograd.set_detect_anomaly(True)

        # Convert device string to torch.device
        args.device = (torch.device(args.device) if torch.cuda.is_available()
                       else torch.device('cpu'))

        # Load model if specified, or test/feature flags are set
        checkpoint = None
        if args.load is not None:
            checkpoint = self.load_checkpoint(args.load, args.device)
        elif args.test:
            # Load best model in output directory if unspecified
            model_path = os.path.join(args.save_dir, "best.pth")
            checkpoint = self.load_checkpoint(model_path, args.device)

        if checkpoint is not None:
            # Use loaded modalities
            args.modalities = checkpoint['modalities']

        # Load data for specified modalities
        self.train_data, self.test_data = self.load_data(args.modalities, args)

        # Resolve short model names to long model names
        args.model = models.names.get(args.model, args.model)

        # Construct model
        if hasattr(models, args.model):
            print('Constructing model...')
            constructor = getattr(models, args.model)
            self.model = self.build_model(constructor, args)
            n_params = sum(p.numel() for p in self.model.parameters()
                           if p.requires_grad)
            print('Number of parameters:', n_params)
        else:
            print('Model name not recognized.')
            return
        
        # Load model state from checkpoint
        if checkpoint is not None:
            self.model.load_state_dict(checkpoint['model'])

        # Fill in unspecified default arguments
        args = self.default_args(args)
            
        # Set up optimizer
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=args.lr, weight_decay=args.w_decay)
        
    def train(self, loader, epoch, args):
        """Train for a single epoch using batched gradient descent."""
        model, optimizer = self.model, self.optimizer
        model.train()
        data_num, loss = 0, 0.0
        rec_mults = dict(args.rec_mults)
        # Iterate over batches
        for b_num, (targets, mask, lengths, _) in enumerate(loader):
            # Anneal KLD loss multipliers
            b_tot = b_num + epoch*len(loader)
            kld_mult =\
                anneal(0.0, args.kld_mult, b_tot, args.kld_anneal*len(loader))
            # Send to device
            mask = mask.to(args.device)
            for m in targets.keys():
                targets[m] = targets[m].to(args.device)
            # Introduce burst deletions to improve interpolation
            inputs = mseq.burst_delete(targets, args.burst_frac, lengths)
            # Compute batch loss
            b_loss = model.step(inputs, mask, kld_mult, rec_mults,
                                targets=targets, lengths=lengths,
                                **args.train_args)
            loss += b_loss
            # Average over number of datapoints before stepping
            b_loss /= sum(lengths)
            b_loss.backward()
            # Plot gradients
            if args.gradients:
                plot_grad_flow(model.named_parameters())
            # Gradient clipping
            if args.clip_grad > 0:
                clip_grad_norm_(model.parameters(), args.clip_grad)
            # Step, then zero gradients
            optimizer.step()
            optimizer.zero_grad()
            # Keep track of total number of time-points
            data_num += sum(lengths)
            print('Batch: {:5d}\tLoss: {:10.1f}'.\
                  format(b_num, loss/data_num))
        # Average losses and print
        loss /= data_num
        print('---')
        print('Epoch: {}\tLoss: {:10.1f}\tKLD-Mult: {:0.3f}'.\
              format(epoch, loss, kld_mult))
        return loss

    def evaluate(self, loader, args):
        model = self.model
        model.eval()
        # Set up accumulators
        n_timesteps = 0
        metrics = None
        results = {'targets': [], 'inputs': [], 'recon': []}
        # Only compute reconstruction loss for specified modalities
        rec_mults = dict(args.rec_mults)
        if args.eval_mods is not None:
            for m in rec_mults:
                rec_mults[m] *= float(m in args.eval_mods)
        # Iterate over batches
        for b_num, (targets, mask, lengths, order) in enumerate(loader):
            # Send to device
            mask = mask.to(args.device)
            for m in targets.keys():
                targets[m] = targets[m].to(args.device)
            # Randomly remove a fraction of observations to test robustness
            inputs = mseq.rand_delete(targets, args.drop_frac, lengths)
            # Remove init/final fraction of observations to test extrapolation
            inputs = mseq.keep_segment(inputs, args.start_frac,
                                       args.stop_frac, lengths)
            # Remove / keep specified modalities to test conditioned generation
            for m in args.drop_mods:
                inputs[m][:] = float('nan')
            for m in args.keep_mods:
                inputs[m] = targets[m].clone().detach()
            # Run forward pass using all modalities, get MAP estimate
            eval_args = {'sample': False}
            eval_args.update(args.eval_args)
            infer, prior, recon = model(inputs, lengths=lengths, **eval_args)
            # Keep track of total number of time-points
            n_timesteps += sum(lengths)
            # Compute and accumulate metrics for this batch
            b_metrics = self.compute_metrics(model, infer, prior, recon,
                                             targets, mask, lengths, order,
                                             args)
            metrics = (b_metrics if metrics is None else
                       {k: metrics[k] + b_metrics[k] for k in metrics})
            # Decollate and store observations and predictions
            results['targets'].\
                append(mseq.seq_decoll_dict(targets, lengths, order))
            results['inputs'].\
                append(mseq.seq_decoll_dict(inputs, lengths, order))
            results['recon'].\
                append(mseq.seq_decoll_dict(recon, lengths, order))
        # Concatenate results across batches
        for k in results:
            modalities = results[k][0].keys()
            results[k] = {m: [seq for batch in results[k] for seq in batch[m]]
                          for m in modalities}
        # Plot predictions against truth
        if args.visualize:
             self.visualize(results, metrics[args.viz_metric], args)
        # Summarize and print metrics
        metrics = self.summarize_metrics(metrics, n_timesteps)
        return results, metrics

    def save_params(self, args):
        """Save training parameters to file."""
        model = self.model
        fname = 'param_hist.tsv'
        df = pd.DataFrame([vars(args)], columns=vars(args).keys())
        df = df[['save_dir', 'model', 'modalities',
                 'batch_size', 'split', 'epochs', 'lr', 'w_decay', 'seed',
                 'burst_frac', 'kld_mult', 'rec_mults', 'kld_anneal',
                 'model_args', 'train_args', 'eval_args']]
        df['h_dim'] = model.h_dim
        df['z_dim'] = model.z_dim
        df.to_csv(fname, mode='a',
                  header=(not os.path.exists(fname)), sep='\t')
    
    def build_model(self, constructor, args):
        raise NotImplementedError
        return model
    
    def load_data(self, modalities, args):
        raise NotImplementedError
        return train_data, test_data

    def default_args(self, args, model):
        """Fill unspecified args with default values."""
        return args
    
    def compute_metrics(self, model, infer, prior, recon,
                        targets, mask, lengths, order, args):
        """Compute evaluation metrics from batch of inputs and outputs."""    
        raise NotImplementedError
        return metrics

    def summarize_metrics(self, metrics, n_timesteps):
        """Summarize and print metrics across dataset."""
        raise NotImplementedError
        return summary

    def visualize(self, results, metric, args):
        """Visualize results."""
        raise NotImplementedError

    def save_results(self, results, args):
        """Save results to file."""
        raise NotImplementedError

    def save_checkpoint(self, modalities, model, path):
        checkpoint = {'modalities': modalities, 'model': model.state_dict()}
        torch.save(checkpoint, path)

    def load_checkpoint(self, path, device):
        checkpoint = torch.load(path, map_location=device)
        return checkpoint
    
    def run_eval(self, args):
        """Evaluate on both training and test set."""
        print("--Training--")
        eval_loader = DataLoader(self.train_data, batch_size=args.batch_size,
                                 collate_fn=mseq.seq_collate_dict,
                                 shuffle=False, pin_memory=True)
        with torch.no_grad():
            args.eval_set = 'train'
            results, _  = self.evaluate(eval_loader, args)
            self.save_results(results, args)

        print("--Testing--")
        eval_loader = DataLoader(self.test_data, batch_size=args.batch_size,
                                 collate_fn=mseq.seq_collate_dict,
                                 shuffle=False, pin_memory=True)
        with torch.no_grad():
            args.eval_set = 'test'
            results, _  = self.evaluate(eval_loader, args)
            self.save_results(results, args)

        # Save command line flags, model params
        self.save_params(args)

    def run_find(self, args):
        """Finds best trained model in save directory."""
        model = self.model
        test_loader = DataLoader(self.test_data, batch_size=args.batch_size,
                                 collate_fn=mseq.seq_collate_dict,
                                 shuffle=False, pin_memory=True)
        best_loss, best_epoch = float('inf'), -1
        args.eval_set = None

        # Iterate over saved models
        for epoch in range(args.save_freq, args.epochs+1, args.save_freq):
            path = os.path.join(args.save_dir,
                                "epoch_{}.pth".format(epoch))
            # Skip epochs without saved models
            if not os.path.exists(path):
                continue
            checkpoint = self.load_checkpoint(path, args.device)
            model.load_state_dict(checkpoint['model'])
            print('--- Epoch {} ---'.format(epoch))
            with torch.no_grad():
                _, metrics = self.evaluate(test_loader, args)
                loss = metrics[args.eval_metric]
            if loss < best_loss:
                best_loss, best_epoch = loss, epoch
                path = os.path.join(args.save_dir, "best.pth")
                self.save_checkpoint(args.modalities, model, path)

        # Print results for best model
        print('=== Best Epoch : {} ==='.format(best_epoch))
        path = os.path.join(args.save_dir, "best.pth")
        checkpoint = self.load_checkpoint(path, args.device)
        model.load_state_dict(checkpoint['model'])
        with torch.no_grad():
            self.evaluate(test_loader, args)

        # Save command line flags, model params
        self.save_params(args)

    def run_train(self, args):
        """Train model over many epochs."""
        train_data, test_data = self.train_data, self.test_data
        
        # Corrupt training data if flags are specified
        if 'uniform' in args.corrupt:
            # Uniform random deletion
            train_data =\
                train_data.corrupt(args.corrupt['uniform'])
        if 'burst' in args.corrupt:
            # Burst deletion
            train_data =\
                train_data.corrupt(args.corrupt['burst'], mode='burst')
        if 'semi' in args.corrupt:
            # Delete entire modalities at random
            train_data =\
                train_data.corrupt(args.corrupt['semi'], mode='all_none',
                                   modalities=args.corrupt['modalities'])
            
        # Split training data into chunks
        train_data = train_data.split(args.split, args.bylen)
        # Batch data using data loaders
        train_loader = DataLoader(train_data, batch_size=args.batch_size,
                                  collate_fn=mseq.seq_collate_dict,
                                  shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_data, batch_size=args.batch_size,
                                 collate_fn=mseq.seq_collate_dict,
                                 shuffle=False, pin_memory=True)

        # Create path to save models/predictions
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        
        # Train and save best model
        best_loss = float('inf')
        args.eval_set = None
        for epoch in range(1, args.epochs + 1):
            print('---')
            self.train(train_loader, epoch, args)
            if epoch % args.eval_freq == 0:
                # Evaluate model every eval_freq epochs
                with torch.no_grad():
                    _, metrics = self.evaluate(test_loader, args)
                    loss = metrics[args.eval_metric]
                # Save model with best metric so far (lower is better)
                if loss < best_loss:
                    best_loss = loss
                    path = os.path.join(args.save_dir, "best.pth") 
                    self.save_checkpoint(args.modalities, self.model, path)
            # Save checkpoints
            if epoch % args.save_freq == 0:
                path = os.path.join(args.save_dir,
                                    "epoch_{}.pth".format(epoch)) 
                self.save_checkpoint(args.modalities, self.model, path)

        # Save final model
        path = os.path.join(args.save_dir, "last.pth") 
        self.save_checkpoint(args.modalities, self.model, path)

        # Save command line flags, model params and performance statistics
        self.save_params(args)
        
    def run(self, args):
        # Evaluate model if test flag is set
        if args.test:
            self.run_eval(args)
            return

        # Find best trained model in save directory
        if args.find_best:
            self.run_find(args)
            return

        # Train model if neither flag is specified
        self.run_train(args)
