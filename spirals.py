"""Training code for the noisy spirals dataset."""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys, os, shutil
import argparse, yaml

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import multiseq as mseq
from datasets.spirals import SpiralsDataset

import models
from utils import eval_ccc, anneal, plot_grad_flow

def train(loader, model, optimizer, epoch, args):
    model.train()
    loss = 0.0
    data_num = 0
    log_freq = len(loader) // args.log_freq
    rec_mults = dict(args.rec_mults)
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
        b_loss = model.step(inputs, mask, kld_mult, rec_mults, targets=targets,
                            lengths=lengths, **args.train_args)
        loss += b_loss
        # Average over number of datapoints before stepping
        b_loss /= sum(lengths)
        b_loss.backward()
        # Plot gradients
        if args.gradients:
            plot_grad_flow(model.named_parameters())
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

def evaluate(loader, model, args):
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
        infer, prior, recon = model(inputs, lengths=lengths, sample=False,
                                    **args.eval_args)
        # Keep track of total number of time-points
        n_timesteps += sum(lengths)
        # Compute and accumulate metrics for this batch
        b_metrics = compute_metrics(model, infer, prior, recon,
                                    targets, mask, lengths, order, args)
        metrics = (b_metrics if metrics is None else
                   {k: metrics[k] + b_metrics[k] for k in metrics})
        # Decollate and store observations and predictions
        results['targets'].append(mseq.seq_decoll_dict(targets,lengths,order))
        results['inputs'].append(mseq.seq_decoll_dict(inputs,lengths,order))
        results['recon'].append(mseq.seq_decoll_dict(recon,lengths,order))
    # Concatenate results across batches
    for k in results:
        modalities = results[k][0].keys()
        results[k] = {m: [seq for batch in results[k] for seq in batch[m]] for
                      m in modalities}
    # Plot predictions against truth
    if args.visualize:
         visualize(results, metrics[args.viz_metric], args)
    # Summarize and print metrics
    metrics = summarize_metrics(metrics, n_timesteps)
    return results, metrics

def compute_metrics(model, infer, prior, recon,
                    targets, mask, lengths, order, args):
    """Compute evaluation metrics from batch of inputs and outputs."""    
    metrics = dict()
    if type(lengths) != torch.tensor:
        lengths = torch.tensor(lengths).float().to(args.device)
    # Compute and store KLD and reconstruction losses
    metrics['kld_loss'] = model.kld_loss(infer, prior, mask)
    metrics['rec_loss'] = model.rec_loss(targets, recon, mask, args.rec_mults)
    # Compute mean squared error in 2D space for each time-step
    mse = sum([(recon[m][0]-targets[m]).pow(2) for m in recon.keys()])
    mse = mse.sum(dim=range(2, mse.dim()))
    # Average across timesteps, for each sequence
    def time_avg(val):
        val[1 - mask.squeeze(-1)] = 0.0
        return val.sum(dim = 0) / lengths
    metrics['mse'] = time_avg(mse)[order].tolist()    
    return metrics
    
def summarize_metrics(metrics, n_timesteps):
    """Summarize and print metrics across dataset."""
    summary = dict()
    for key, val in metrics.items():
        if type(val) is list:
            # Compute mean and std dev. of metric over sequences
            summary[key] = np.mean(val)
            summary[key + '_std'] = np.std(val)
        else:
            # Average over all timesteps
            summary[key] = val / n_timesteps
    print('Evaluation\tKLD: {:7.1f}\tRecon: {:7.1f}\t  MSE: {:6.3f} +-{:2.3f}'\
          .format(summary['kld_loss'], summary['rec_loss'],
                  summary['mse'], summary['mse_std']))
    return summary

def visualize(results, metric, args):
    """Plots predictions against truth for representative fits."""
    reference = results['targets']
    observed = results['inputs']
    predicted = results['recon']

    # Select top 4 and bottom 4 predictions
    sel_idx = np.concatenate((np.argsort(metric)[:4],
                              np.argsort(metric)[-4:][::-1]))
    sel_metric = [metric[i] for i in sel_idx]
    sel_true = [reference['metadata'][i][:,0:2] for i in sel_idx]
    sel_true = [(arr[:,0], arr[:,1]) for arr in sel_true]
    sel_data = [(reference['spiral-x'][i], reference['spiral-y'][i])
                for i in sel_idx]
    sel_obsv = [(observed['spiral-x'][i], observed['spiral-y'][i])
               for i in sel_idx]
    sel_pred = [(predicted['spiral-x'][i][:,0], predicted['spiral-y'][i][:,0])
                for i in sel_idx]
    sel_rng = [(predicted['spiral-x'][i][:,1], predicted['spiral-y'][i][:,1])
                for i in sel_idx]

    # Create figure to visualize predictions
    if not hasattr(args, 'fig'):
        args.fig, args.axes =\
            plt.subplots(4, 2, figsize=(4,8), subplot_kw={'aspect': 'equal'})
    else:
        plt.figure(args.fig.number)
    axes = args.axes
    
    # Set current figure
    plt.figure(args.fig.number)
    for i in range(len(sel_idx)):
        axis = args.axes[(i % 4),(i // 4)]
        # Plot spiral
        plot_spiral(axis, sel_true[i], sel_data[i],
                    sel_obsv[i], sel_pred[i], sel_rng[i])
        # Set title as metric
        axis.set_title("Metric = {:0.3f}".format(sel_metric[i]))
        axis.set_xlabel("Spiral {:03d}".format(sel_idx[i]))
        
    plt.tight_layout()
    plt.draw()
    if args.eval_set is not None:
        fig_path = os.path.join(args.save_dir, args.eval_set + '.pdf')
        plt.savefig(fig_path)
    plt.pause(1.0 if args.test else 0.001)

def plot_spiral(axis, true, data, obsv, pred, rng):
    axis.cla()
    # Plot 95% confidence ellipses
    ec = EllipseCollection(1.96*rng[0], 1.96*rng[1], (0,), units='x',
                           facecolors=('c',), alpha=0.25,
                           offsets=np.column_stack(pred),
                           transOffset=axis.transData)
    axis.add_collection(ec)

    # Plot ground truth
    axis.plot(true[0], true[1], 'b-', linewidth=1.5)

    # Plot observations (blue = both, pink = x-only, yellow = y-only)
    if (np.isnan(obsv[0]) != np.isnan(obsv[1])).any():
        axis.plot(obsv[0], data[1], '<', markersize=2, color='#fe46a5')
        axis.plot(data[0], obsv[1], 'v', markersize=2, color='#fec615')
    axis.plot(obsv[0], obsv[1], 'bo', markersize=3)

    # Plot predictions
    axis.plot(pred[0], pred[1], '-', linewidth=1.5, color='#04d8b2')

    # Set limits
    axis.set_xlim(-4, 4)
    axis.set_ylim(-4, 4)

def save_results(results, args):
    pass
    
def save_params(args, model):
    fname = 'param_hist.tsv'
    df = pd.DataFrame([vars(args)], columns=vars(args).keys())
    df = df[['save_dir', 'model', 'modalities',
             'batch_size', 'split', 'epochs', 'lr', 'w_decay', 'seed',
             'burst_frac', 'kld_mult', 'rec_mults', 'kld_anneal',
             'model_args', 'train_args', 'eval_args']]
    df['h_dim'] = model.h_dim
    df['z_dim'] = model.z_dim
    df.to_csv(fname, mode='a', header=(not os.path.exists(fname)), sep='\t')
        
def save_checkpoint(modalities, model, path):
    checkpoint = {'modalities': modalities, 'model': model.state_dict()}
    torch.save(checkpoint, path)

def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    return checkpoint

def load_data(modalities, args):
    print("Loading data...")
    data_dir = os.path.abspath(args.data_dir)
    train_data = SpiralsDataset(modalities, data_dir, args.train_subdir,
                                truncate=True, item_as_dict=True)
    test_data = SpiralsDataset(modalities, data_dir, args.test_subdir,
                               truncate=True, item_as_dict=True)
    print("Done.")
    if len(args.normalize) > 0:
        print("Normalizing ", args.normalize, "...")
        # Normalize test data using training data as reference
        test_data.normalize_(modalities=args.normalize, ref_data=train_data)
        # Normalize training data in-place
        train_data.normalize_(modalities=args.normalize)
    return train_data, test_data

def main(args):
    # Fix random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Convert device string to torch.device
    args.device = (torch.device(args.device) if torch.cuda.is_available()
                   else torch.device('cpu'))

    # Load model if specified, or test/feature flags are set
    checkpoint = None
    if args.load is not None:
        checkpoint = load_checkpoint(args.load, args.device)
    elif args.test:
        # Load best model in output directory if unspecified
        model_path = os.path.join(args.save_dir, "best.pth")
        checkpoint = load_checkpoint(model_path, args.device)
    
    if checkpoint is not None:
        # Use loaded modalities
        args.modalities = checkpoint['modalities']
    elif args.modalities is None:
        # Default to all if unspecified
        args.modalities = ['spiral-x', 'spiral-y']

    # Load data for specified modalities
    train_data, test_data = load_data(args.modalities, args)

    # Resolve short model names to long model names
    args.model = models.names.get(args.model, args.model)

    # Construct model
    dims = {'spiral-x': 1, 'spiral-y': 1}
    if hasattr(models, args.model):
        print('Constructing model...')
        constructor = getattr(models, args.model)
        model = constructor(args.modalities,
                            dims=(dims[m] for m in args.modalities),
                            z_dim=5, h_dim=20,
                            device=args.device, **args.model_args)
        n_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
        print('Number of parameters:', n_params)
    else:
        print('Model name not recognized.')
        return
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])

    # Default reconstruction loss multipliers
    if args.rec_mults is None:
        args.rec_mults = {m : (1.0 / dims[m]) / len(args.modalities)
                          for m in args.modalities}
        
    # Setup loss and optimizer
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.w_decay)

    # Create path to save models/predictions
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    # Create figure to visualize predictions
    if args.visualize:
        args.fig, args.axes = plt.subplots(4, 2, figsize=(4,8),
                                           subplot_kw={'aspect': 'equal'})
        
    # Evaluate model if test flag is set
    if args.test:            
        # Evaluate on both training and test set        
        print("--Training--")
        eval_loader = DataLoader(train_data, batch_size=args.batch_size,
                                 collate_fn=mseq.seq_collate_dict,
                                 shuffle=False, pin_memory=True)
        with torch.no_grad():
            args.eval_set = 'train'
            results, _  = evaluate(eval_loader, model, args)
            save_results(results, args)
            
        print("--Testing--")
        eval_loader = DataLoader(test_data, batch_size=args.batch_size,
                                 collate_fn=mseq.seq_collate_dict,
                                 shuffle=False, pin_memory=True)
        with torch.no_grad():
            args.eval_set = 'test'
            results, _  = evaluate(eval_loader, model, args)
            save_results(results, args)

        # Save command line flags, model params
        save_params(args, model)
        return

    # Split training data into chunks
    train_data = train_data.split(args.split)
    # Batch data using data loaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              collate_fn=mseq.seq_collate_dict,
                              shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size,
                             collate_fn=mseq.seq_collate_dict,
                             shuffle=False, pin_memory=True)
   
    # Train and save best model
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        print('---')
        train(train_loader, model, optimizer, epoch, args)
        if epoch % args.eval_freq == 0:
            with torch.no_grad():
                _, metrics = evaluate(test_loader, model, args)
                loss = metrics[args.eval_metric]
            if loss < best_loss:
                best_loss = loss
                path = os.path.join(args.save_dir, "best.pth") 
                save_checkpoint(args.modalities, model, path)
        # Save checkpoints
        if epoch % args.save_freq == 0:
            path = os.path.join(args.save_dir, "epoch_{}.pth".format(epoch)) 
            save_checkpoint(args.modalities, model, path)

    # Save final model
    path = os.path.join(args.save_dir, "last.pth") 
    save_checkpoint(args.modalities, model, path)

    # Save command line flags, model params and performance statistics
    save_params(args, model)
    
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='dmm', metavar='S',
                        help='name of model to train (default: dmm)')
    parser.add_argument('--model_args', type=yaml.safe_load, default=dict(),
                        help='additional model arguments as yaml dict')
    parser.add_argument('--train_args', type=yaml.safe_load, default=dict(),
                        help='additional training arguments as yaml dict')
    parser.add_argument('--eval_args', type=yaml.safe_load, default=dict(),
                        help='additional evaluation arguments as yaml dict')
    parser.add_argument('--modalities', type=str, default=None, nargs='+',
                        help='input modalities (default: all')
    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--split', type=int, default=1, metavar='N',
                        help='sections to split each video into (default: 1)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--w_decay', type=float, default=1e-4, metavar='F',
                        help='Adam weight decay (default: 1e-4)')
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                        help='random seed (default: 1)')
    parser.add_argument('--kld_mult', type=float, default=1.0, metavar='F',
                        help='max kld loss multiplier (default: 1.0)')
    parser.add_argument('--rec_mults', type=yaml.safe_load, default=None,
                        help='reconstruction loss multiplier (default: 1/dim)')
    parser.add_argument('--kld_anneal', type=int, default=100, metavar='N',
                        help='epochs to increase kld_mult over (default: 100)')
    parser.add_argument('--burst_frac', type=float, default=0.1, metavar='F',
                        help='burst error rate during training (default: 0.1)')
    parser.add_argument('--drop_frac', type=float, default=0.5, metavar='F',
                        help='fraction of data to randomly drop at test time')
    parser.add_argument('--start_frac', type=float, default=0.25, metavar='F',
                        help='fraction of test trajectory to begin at')
    parser.add_argument('--stop_frac', type=float, default=0.75, metavar='F',
                        help='fraction of test trajectory to stop at')
    parser.add_argument('--drop_mods', type=str, default=[], nargs='+',
                        help='modalities to delete at test (default: none')
    parser.add_argument('--keep_mods', type=str, default=[], nargs='+',
                        help='modalities to retain at test (default: none')
    parser.add_argument('--eval_mods', type=str, default=None, nargs='+',
                        help='modalities to evaluate at test (default: none')
    parser.add_argument('--eval_metric', type=str, default='mse',
                        help='metric to track best model (default: mse)')
    parser.add_argument('--viz_metric', type=str, default='mse',
                        help='metric for visualization (default: mse)')
    parser.add_argument('--log_freq', type=int, default=5, metavar='N',
                        help='print loss N times every epoch (default: 5)')
    parser.add_argument('--eval_freq', type=int, default=10, metavar='N',
                        help='evaluate every N epochs (default: 10)')
    parser.add_argument('--save_freq', type=int, default=10, metavar='N',
                        help='save every N epochs (default: 10)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device to use (default: cuda:0 if available)')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='flag to visualize predictions (default: false)')
    parser.add_argument('--gradients', action='store_true', default=False,
                        help='flag to plot gradients (default: false)')
    parser.add_argument('--normalize', type=str, default=[], nargs='+',
                        help='modalities to normalize (default: [])')
    parser.add_argument('--test', action='store_true', default=False,
                        help='evaluate without training (default: false)')
    parser.add_argument('--load', type=str, default=None,
                        help='path to trained model (either resume or test)')
    parser.add_argument('--data_dir', type=str, default="./datasets/spirals",
                        help='path to data base directory')
    parser.add_argument('--save_dir', type=str, default="./spirals_save",
                        help='path to save models and predictions')
    parser.add_argument('--train_subdir', type=str, default='train',
                        help='training data subdirectory')
    parser.add_argument('--test_subdir', type=str, default='test',
                        help='testing data subdirectory')
    args = parser.parse_args()
    main(args)
