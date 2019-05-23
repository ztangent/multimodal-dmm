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
    mse_std = np.std(mse_loss)
    mse_loss = sum(mse_loss) / len(mse_loss)
    losses = kld_loss, rec_loss, mse_loss
    print('Evaluation\tKLD: {:7.1f}\tRecon: {:7.1f}\t  MSE: {:6.3f} +-{:2.3f}'\
          .format(kld_loss, rec_loss, mse_loss, mse_std))
    return predictions, losses

def eval_suite(item, truth, model, args):
    # Run suite of evaluation tasks on provided item
    model.eval()
    # Collate item to batch tensor
    targets, mask, lengths, order = mseq.seq_collate_dict([item])
    # Send to device
    mask = mask.to(args.device)
    for m in targets.keys():
        targets[m] = targets[m].to(args.device)

    # Process inputs for each evaluation task
    tasks = ['Recon.', 'Drop Half', 'Fwd. Extra.', 'Bwd. Extra.', 'Cond. Gen']
    inputs = dict()
    # For reconstruction provide complete inputs
    inputs[tasks[0]] = {m: targets[m].clone().detach() for m in targets}
    # For drop-half, randomly remove 50% of data
    inputs[tasks[1]] = mseq.rand_delete(targets, 0.5, lengths)
    # For forward extrapolation, remove last 25% of data
    inputs[tasks[2]] = mseq.keep_segment(targets, 0.0, 0.75, lengths)
    # For backward extrapolation, remove first 25% of data
    inputs[tasks[3]] = mseq.keep_segment(targets, 0.25, 1.0, lengths)
    # For conditional generation, remove last 75% of y-coordinate
    inputs[tasks[4]] =\
        mseq.keep_segment(targets, 0.0, 0.25, lengths, ['spiral-y'])
    inputs[tasks[4]]['spiral-x'] = targets['spiral-x'].clone().detach()

    # Create figure and axes
    fig, axes = plt.subplots(1, 5, figsize=(10,2.5),
                             subplot_kw={'aspect': 'equal'})
    
    # Iterate over tasks
    for i, task in enumerate(tasks):
        # Run forward pass observed inputs
        observed = inputs[task]
        infer, prior, recon = model(observed, lengths=lengths,
                                    sample=False, **args.eval_args)
        # Compute KLD and reconstruction losses
        kld_loss = model.kld_loss(infer, prior, mask)
        kld_loss /= lengths[0]
        rec_loss = model.rec_loss(targets, recon, mask, args.rec_mults)
        rec_loss /= lengths[0]
        # Compute mean squared error for each timestep
        mse = sum([(recon[m][0]-targets[m]).pow(2) for m in recon.keys()])
        mse = mse.sum(dim=range(2, mse.dim()))
        # Average across timesteps, for each sequence
        mse = mse.sum(dim=0).cpu() / torch.tensor(lengths).float()
        mse = mse.item()
        # Print losses
        print('{:15} KLD: {:7.1f}\tRecon: {:7.1f}\t  MSE: {:6.3f}'\
              .format(task, kld_loss, rec_loss, mse))
        # Extract quantities for plotting
        data = (targets['spiral-x'][:,0].cpu().numpy(),
                targets['spiral-y'][:,0].cpu().numpy())
        obs = (observed['spiral-x'][:,0].cpu().numpy(),
               observed['spiral-y'][:,0].cpu().numpy())
        pred = (recon['spiral-x'][0][:,0].cpu().numpy(),
                recon['spiral-y'][0][:,0].cpu().numpy())
        rng = (1.96*recon['spiral-x'][1][:,0].cpu().numpy(),
               1.96*recon['spiral-y'][1][:,0].cpu().numpy())
        # Plot spiral
        plot_spiral(axes[i], truth, data, obs, pred, rng)
        axes[i].set_title(task)
        axes[i].set_xticks([], [])
        axes[i].set_yticks([], [])
        for spine in axes[i].spines.values():
            spine.set_visible(False)        
        axes[i].set_xlabel("MSE: {:0.3f}".format(mse))
        if i == 0:
            axes[i].set_ylabel(args.model)
        
    # Display and save figure
    plt.draw()
    plt.savefig('suite.pdf')
    plt.show()

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
        axis = args.axes[(i % 4),(i // 4)]
        # Plot spiral
        plot_spiral(axis, sel_truth[i], sel_data[i],
                    sel_obs[i], sel_pred[i], sel_rng[i])
        # Set title as metric
        axis.set_title("Metric = {:0.3f}".format(sel_metric[i]))
        axis.set_xlabel("Spiral {:03d}".format(sel_idx[i]))
        
    plt.tight_layout()
    plt.draw()
    if fig_path is not None:
        plt.savefig(fig_path)
    plt.pause(1.0 if args.test else 0.001)

def plot_spiral(axis, truth, data, obs, pred, rng):
    axis.cla()
    # Plot confidence ellipses
    ec = EllipseCollection(rng[0], rng[1], (0,), units='x',
                           facecolors=('c',), alpha=0.25,
                           offsets=np.column_stack(pred),
                           transOffset=axis.transData)
    axis.add_collection(ec)

    # Plot ground truth
    axis.plot(truth[0], truth[1], 'b-', linewidth=1.5)

    # Plot observations (blue = both, pink = x-only, yellow = y-only)
    if (np.isnan(obs[0]) != np.isnan(obs[1])).any():
        axis.plot(obs[0], data[1], '<', markersize=2, color='#fe46a5')
        axis.plot(data[0], obs[1], 'v', markersize=2, color='#fec615')
    axis.plot(obs[0], obs[1], 'bo', markersize=3)

    # Plot predictions
    axis.plot(pred[0], pred[1], '-', linewidth=1.5, color='#04d8b2')

    # Set limits
    axis.set_xlim(-4, 4)
    axis.set_ylim(-4, 4)
    
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
                            z_dim=5, h_dim=20,
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

    # Run task suite on data sequence, if given
    if args.suite is not None:
        # Get data item and corresponding ground truth from dataset
        suite_idx = args.suite
        item = test_data[suite_idx]
        truth = test_data.orig['metadata'][suite_idx][:,0:2]
        truth = (truth[:,0], truth[:,1])
        print("Running suite of inference tasks...")
        print("--")
        with torch.no_grad():
            eval_suite(item, truth, model, args)
        return
        
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
    parser.add_argument('--suite', type=int, default=None, metavar='I',
                        help='runs inference suite on spiral I if set')
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
