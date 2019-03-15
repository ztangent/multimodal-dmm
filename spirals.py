"""Training code for VRNN model."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys, os, shutil
import argparse

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.spirals import SpiralsDataset
from datasets.multiseq import seq_collate_dict

from models import MultiVRNN
from utils import eval_ccc, anneal

def train(loader, model, optimizer, epoch, args):
    model.train()
    loss = 0.0
    data_num = 0
    log_freq = len(loader) // args.log_freq
    rec_mults = dict(args.rec_mults)
    for batch_num, (data, mask, lengths) in enumerate(loader):
        # Anneal KLD loss multipliers
        batch_tot = batch_num + epoch*len(loader)
        kld_mult =\
            anneal(0.0, args.kld_mult, batch_tot, args.kld_anneal*len(loader))
        # Send to device
        mask = mask.to(args.device)
        for m in data.keys():
            data[m] = data[m].to(args.device)
        # Compute ELBO loss for individual modalities
        batch_loss = 0
        for m in args.modalities:
            # Run forward pass with modality m
            infer, prior, outputs = model({m: data[m]}, lengths)
            # Compute ELBO loss for modality m
            batch_loss += model.loss({m: data[m]}, infer, prior, outputs, mask,
                                     kld_mult, rec_mults)
        if len(args.modalities) > 1:
            # Run forward pass with all modalities
            infer, prior, outputs = model(data, lengths)
            # Compute ELBO loss for all modalities
            batch_loss += model.loss(data, infer, prior, outputs, mask,
                                     kld_mult, rec_mults)
        loss += batch_loss
        # Average over number of datapoints before stepping
        batch_loss /= sum(lengths)
        batch_loss.backward()
        # Step, then zero gradients
        optimizer.step()
        optimizer.zero_grad()
        # Keep track of total number of time-points
        data_num += sum(lengths)
        print('Batch: {:5d}\tLoss: {:10.1f}'.\
              format(batch_num, loss/data_num))
    # Average losses and print
    loss /= data_num
    print('---')
    print('Epoch: {}\tLoss: {:10.1f}\tKLD-Mult: {:0.3f}'.\
          format(epoch, loss, kld_mult))
    return loss

def evaluate(dataset, model, args, fig_path=None):
    model.eval()
    predictions = {m: [] for m in args.modalities}
    data_num = 0
    kld_loss, rec_loss = [], []
    for data in dataset:
        # Collate data into batch dictionary of size 1
        data, mask, lengths = seq_collate_dict([data])
        # Send to device
        mask = mask.to(args.device)
        for m in data.keys():
            data[m] = data[m].to(args.device)
        # Mask out some data to test for robustness
        inputs = {m: torch.tensor(data[m]) for m in data.keys()}
        for m in inputs.keys():
            # Randomly remove a fraction of observations
            drop_n = int(args.drop_frac * max(lengths))
            drop_idx = np.random.choice(max(lengths), drop_n, False)
            inputs[m][drop_idx,:,:] = float('nan')
            # Remove final fraction of observations to test extrapolation
            keep_n = int(args.keep_frac * max(lengths))
            inputs[m][keep_n:,:,:] = float('nan')
        # Run forward pass using all modalities
        infer, prior, outputs = model(inputs, lengths)
        # Compute and store KLD and reconstruction losses
        kld_loss.append(model.kld_loss(infer, prior, mask))
        rec_loss.append(model.rec_loss(data, outputs, mask, args.rec_mults))
        # Keep track of total number of time-points
        data_num += sum(lengths)
        # Store predictions
        out_mean, _ = outputs
        for m in out_mean.keys():
            predictions[m].append(out_mean[m].view(-1).cpu().numpy())
    # Plot predictions against truth
    if args.visualize:
        plot_predictions(dataset, predictions, rec_loss, args, fig_path)
    # Average losses and print
    kld_loss = sum(kld_loss) / data_num
    rec_loss = sum(rec_loss) / data_num
    losses = kld_loss, rec_loss
    print('Evaluation\tKLD: {:7.1f}\tRecon: {:7.1f}'.format(*losses))
    return predictions, losses

def plot_predictions(dataset, predictions, metric, args, fig_path=None):
    """Plots predictions against truth for representative fits."""
    # Select top 4 and bottom 4
    sel_idx = np.concatenate((np.argsort(metric)[:4],
                              np.argsort(metric)[-4:][::-1]))
    sel_metric = [metric[i] for i in sel_idx]
    sel_true = [(dataset.orig['spiral-x'][i], dataset.orig['spiral-y'][i])
                for i in sel_idx]
    sel_pred = [(predictions['spiral-x'][i], predictions['spiral-y'][i])
                for i in sel_idx]
    for i, (true, pred, m) in enumerate(zip(sel_true, sel_pred, sel_metric)):
        j, i = (i // 4), (i % 4)
        args.axes[i,j].cla()
        args.axes[i,j].plot(true[0], true[1], 'b-')
        args.axes[i,j].plot(pred[0], pred[1], 'c-')
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
    df = df[['save_dir', 'modalities', 'normalize', 'batch_size', 'split',
             'epochs', 'lr', 'kld_mult', 'rec_mults',
             'kld_anneal', 'base_rate']]
    df.insert(0, 'model', [model.__class__.__name__])
    df['h_dim'] = model.h_dim
    df['z_dim'] = model.z_dim
    df.set_index('model')
    df.to_csv(fname, mode='a', header=(not os.path.exists(fname)), sep='\t')
        
def save_checkpoint(modalities, model, path):
    checkpoint = {'modalities': modalities, 'model': model.state_dict()}
    torch.save(checkpoint, path)

def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    return checkpoint

def load_data(modalities, args):
    print("Loading data...")
    train_data = SpiralsDataset(modalities, args.data_dir, args.train_subdir,
                                base_rate=args.base_rate,
                                truncate=True, item_as_dict=True)
    test_data = SpiralsDataset(modalities, args.data_dir, args.test_subdir,
                               base_rate=args.base_rate,
                               truncate=True, item_as_dict=True)
    print("Done.")
    if len(args.normalize) > 0:
        print("Normalizing ", args.normalize, "...")
        # Normalize test data using training data as reference
        test_data.normalize_(modalities=args.normalize, ref_data=train_data)
        # Normailze training data in-place
        train_data.normalize_(modalities=args.normalize)
    return train_data, test_data

def main(args):
    # Fix random seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)

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
    
    # Construct model
    dims = {'spiral-x': 1, 'spiral-y': 1}
    if args.model == 'MultiVRNN':
        model = MultiVRNN(args.modalities,
                          dims=(dims[m] for m in args.modalities),
                          device=args.device)
    else:
        print('Model name not recognized'.)
        return
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])

    # Default reconstruction loss multipliers
    if args.rec_mults is None:
        args.rec_mults = {m : (1.0 / dims[m]) / len(args.modalities)
                          for m in args.modalities}
        
    # Setup loss and optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Create path to save models/predictions
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Create figure to visualize predictions
    if args.visualize:
        args.fig, args.axes = plt.subplots(4, 2, figsize=(6,8))
        
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
        with torch.no_grad():
            print("--Training--")
            pred, _  = evaluate(train_data, model, args,
                                os.path.join(args.save_dir, "train.png"))
            # save_predictions(train_data, pred, pred_train_dir)
            print("--Testing--")
            pred, _  = evaluate(test_data, model, args,
                                os.path.join(args.save_dir, "test.png"))
            # save_predictions(test_data, pred, pred_test_dir)
        # Save command line flags, model params
        save_params(args, model)
        return

    # Split training data into chunks
    train_data = train_data.split(args.split)
    # Batch data using data loaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=True, collate_fn=seq_collate_dict,
                              pin_memory=True)
   
    # Train and save best model
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        print('---')
        train(train_loader, model, optimizer, epoch, args)
        if epoch % args.eval_freq == 0:
            with torch.no_grad():
                pred, (kld_loss, rec_loss) = evaluate(test_data, model, args)
            if rec_loss < best_loss:
                best_loss = rec_loss
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
    parser.add_argument('--model', type=str, default='MultiVRNN', metavar='S',
                        help='name of model to train (default: MultiVRNN)')
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
    parser.add_argument('--base_rate', type=float, default=1.0, metavar='N',
                        help='sampling rate to resample to (default: 1.0)')
    parser.add_argument('--kld_mult', type=float, default=1.0, metavar='F',
                        help='max kld loss multiplier (default: 1.0)')
    parser.add_argument('--rec_mults', type=float, default=None, nargs='+',
                        help='reconstruction loss multiplier (default: 1/dims')
    parser.add_argument('--kld_anneal', type=int, default=100, metavar='N',
                        help='epochs to increase kld_mult over (default: 100)')
    parser.add_argument('--drop_frac', type=float, default=0.5, metavar='F',
                        help='fraction of data to randomly drop at test time')
    parser.add_argument('--keep_frac', type=float, default=0.75, metavar='F',
                        help='fraction of trajectory to keep at test time')
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
