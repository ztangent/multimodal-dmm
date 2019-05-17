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

def evaluate(loader, model, args, fig_path=None):
    model.eval()
    predictions = {m: [] for m in model.modalities}
    observed = {m: [] for m in model.modalities}
    ranges = {m: [] for m in model.modalities}
    data_num = 0
    kld_loss, rec_loss, mse_loss = [], [], []
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
        # Run forward pass using all modalities, get MAP estimate
        infer, prior, recon = model(inputs, lengths=lengths, sample=False,
                                    **args.eval_args)
        # Compute and store KLD and reconstruction losses
        kld_loss.append(model.kld_loss(infer, prior, mask))
        rec_loss.append(model.rec_loss(targets, recon, mask, args.rec_mults))
        # Keep track of total number of time-points
        data_num += sum(lengths)
        # Decollate and store observations and predictions
        for m in recon.keys():
            observed[m] += mseq.seq_decoll(inputs[m], lengths, order)
            predictions[m] += mseq.seq_decoll(recon[m][0], lengths, order)
            ranges[m] += mseq.seq_decoll(1.96 * recon[m][1], lengths, order)
        # Compute mean squared error for each timestep
        mse = sum([(recon[m][0]-targets[m]).pow(2) for m in recon.keys()])
        mse = mse.sum(dim=range(2, mse.dim()))
        # Average across timesteps, for each sequence
        mse[1 - mask.squeeze(-1)] = 0.0
        mse = mse.sum(dim=0).cpu() / torch.tensor(lengths).float()
        mse_loss += mse[order].tolist()
    # Plot predictions against truth
    if args.visualize:
        visualize(loader.dataset, observed, predictions, ranges,
                  mse_loss, args, fig_path)
    # Average losses and print
    kld_loss = sum(kld_loss) / data_num
    rec_loss = sum(rec_loss) / data_num
    mse_loss = sum(mse_loss) / len(loader.dataset)
    losses = kld_loss, rec_loss, mse_loss
    print('Evaluation\tKLD: {:7.1f}\tRecon: {:7.1f}\t  MSE: {:6.3f}'\
          .format(kld_loss, rec_loss, mse_loss))
    return predictions, losses

def visualize(dataset, observed, predictions, ranges,
              metric, args, fig_path=None):
    """Plots predictions against truth for representative fits."""
    # Select top 4 and bottom 4
    sel_idx = np.concatenate((np.argsort(metric)[:4],
                              np.argsort(metric)[-4:][::-1]))
    sel_metric = [metric[i] for i in sel_idx]
    sel_truth = [dataset.orig['metadata'][i][:,0:2] for i in sel_idx]
    sel_truth = [(arr[:,0], arr[:,1]) for arr in sel_truth]
    sel_data = [(dataset.orig['spiral-x'][i], dataset.orig['spiral-y'][i])
                for i in sel_idx]
    sel_obs = [(observed['spiral-x'][i], observed['spiral-y'][i])
               for i in sel_idx]
    sel_pred = [(predictions['spiral-x'][i], predictions['spiral-y'][i])
                for i in sel_idx]
    sel_rng = [(ranges['spiral-x'][i], ranges['spiral-y'][i])
               for i in sel_idx]

    # Set current figure
    plt.figure(args.fig.number)
    for i in range(len(sel_idx)):
        truth, data = sel_truth[i], sel_data[i]
        obs, pred = sel_obs[i], sel_pred[i]
        rng, m = sel_rng[i], sel_metric[i]
        j, i = (i // 4), (i % 4)
        args.axes[i,j].cla()

        # Plot confidence ellipses
        ec = EllipseCollection(rng[0], rng[1], (0,), units='x',
                               facecolors=('c',), alpha=0.25,
                               offsets=np.column_stack(pred),
                               transOffset=args.axes[i,j].transData)
        args.axes[i,j].add_collection(ec)
        
        # Plot ground truth
        args.axes[i,j].plot(truth[0], truth[1], 'b-', linewidth=1)

        # Plot observations (blue = both, magenta = x-only, yellow = y-only)
        if (np.isnan(obs[0]) != np.isnan(obs[1])).any():
            args.axes[i,j].plot(obs[0], data[1], 'm.', markersize=1.5)
            args.axes[i,j].plot(data[0], obs[1], 'y.', markersize=1.5)
        args.axes[i,j].plot(obs[0], obs[1], 'b.', markersize=1.5)

        # Plot predictions
        args.axes[i,j].plot(pred[0], pred[1], 'g-', linewidth=1)
        
        # Set limits and title
        args.axes[i,j].set_xlim(-5, 5)
        args.axes[i,j].set_ylim(-5, 5)
        args.axes[i,j].set_title("Metric = {:0.3f}".format(m))
        
    plt.tight_layout()
    plt.draw()
    if fig_path is not None:
        plt.savefig(fig_path)
    plt.pause(1.0 if args.test else 0.001)

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
        constructor = getattr(models, args.model)
        model = constructor(args.modalities,
                            dims=(dims[m] for m in args.modalities),
                            device=args.device, **args.model_args)
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
        # Create paths to save predictions
        pred_train_dir = os.path.join(args.save_dir, "pred_train")
        pred_test_dir = os.path.join(args.save_dir, "pred_test")
        if not os.path.exists(pred_train_dir):
            os.makedirs(pred_train_dir)
        if not os.path.exists(pred_test_dir):
            os.makedirs(pred_test_dir)
            
        # Evaluate on both training and test set        
        print("--Training--")
        eval_loader = DataLoader(train_data, batch_size=args.batch_size,
                                 collate_fn=mseq.seq_collate_dict,
                                 shuffle=False, pin_memory=True)
        with torch.no_grad():
            pred, _  = evaluate(eval_loader, model, args,
                                os.path.join(args.save_dir, "train.pdf"))
            # save_predictions(train_data, pred, pred_train_dir)
            
        print("--Testing--")
        eval_loader = DataLoader(test_data, batch_size=args.batch_size,
                                 collate_fn=mseq.seq_collate_dict,
                                 shuffle=False, pin_memory=True)
        with torch.no_grad():
            pred, _  = evaluate(eval_loader, model, args,
                                os.path.join(args.save_dir, "test.pdf"))
            # save_predictions(test_data, pred, pred_test_dir)

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
                pred, losses = evaluate(test_loader, model, args)
                _, _, loss = losses
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
    parser.add_argument('--model', type=str, default='bdmm', metavar='S',
                        help='name of model to train (default: bdmm)')
    parser.add_argument('--model_args', type=yaml.load, default=dict(),
                        help='additional model arguments as yaml dict')
    parser.add_argument('--train_args', type=yaml.load, default=dict(),
                        help='additional training arguments as yaml dict')
    parser.add_argument('--eval_args', type=yaml.load, default=dict(),
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
    parser.add_argument('--rec_mults', type=yaml.load, default=None,
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
