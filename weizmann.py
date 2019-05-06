"""Training code for the Weizmann human action dataset."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys, os, shutil
import argparse, yaml

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import multiseq as mseq
from datasets.weizmann import WeizmannDataset

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
        inputs = mseq.burst_delete(targets, args.burst_frac)
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
    predictions = {m: [] for m in args.modalities}
    observed = {m: [] for m in args.modalities}
    data_num = 0
    kld_loss, rec_loss, mse_loss = [], [], []
    for b_num, (targets, mask, lengths, order) in enumerate(loader):
        # Send to device
        mask = mask.to(args.device)
        for m in targets.keys():
            targets[m] = targets[m].to(args.device)
        # Randomly remove a fraction of observations to test robustness
        inputs = mseq.rand_delete(targets, args.drop_frac)
        # Remove init/final fraction of observations to test extrapolation
        t_start = int(args.start_frac * max(lengths))
        t_stop = int(args.stop_frac * max(lengths))
        inputs = mseq.keep_segment(inputs, t_start, t_stop)
        # Run forward pass using all modalities, get MAP estimate
        infer, prior, outputs = model(inputs, lengths=lengths, sample=False,
                                      **args.eval_args)
        # Compute and store KLD and reconstruction losses
        kld_loss.append(model.kld_loss(infer, prior, mask))
        rec_loss.append(model.rec_loss(targets, outputs, mask, args.rec_mults))
        # Keep track of total number of time-points
        data_num += sum(lengths)
        # Decollate and store observations and predictions
        out_mean, out_std = outputs
        for m in out_mean.keys():
            observed[m] += mseq.seq_decoll(inputs[m], lengths, order)
            predictions[m] += mseq.seq_decoll(out_mean[m], lengths, order)
        # Compute mean squared error for each timestep
        mse = sum([(out_mean[m] - targets[m]).pow(2) / out_mean[m][0,0].numel()
                   for m in out_mean.keys()])
        mse = mse.sum(dim=range(2, mse.dim()))
        # Average across timesteps, for each sequence
        mse[1 - mask.squeeze(-1)] = 0.0
        mse = mse.sum(dim=0).cpu() / torch.tensor(lengths).float()
        mse_loss += mse[order].tolist()
    # Plot predictions against truth
    if args.visualize:
         visualize(loader.dataset, observed, predictions,
                   mse_loss, args, fig_path)
    # Average losses and print
    kld_loss = sum(kld_loss) / data_num
    rec_loss = sum(rec_loss) / data_num
    mse_loss = sum(mse_loss) / len(loader.dataset)
    losses = kld_loss, rec_loss, mse_loss
    print('Evaluation\tKLD: {:7.1f}\tRecon: {:7.1f}\t  MSE: {:6.3f}'\
          .format(kld_loss, rec_loss, mse_loss))
    return predictions, losses

def visualize(dataset, observed, predictions, metric, args, fig_path=None):
    """Plots predictions against truth for representative fits."""
    # Select best and worst predictions
    sel_idx = np.concatenate((np.argsort(metric)[:1],
                              np.argsort(metric)[-1:][::-1]))
    sel_metric = [metric[i] for i in sel_idx]
    sel_truth = [dataset[i]['video'] for i in sel_idx]
    sel_obs = [observed['video'][i] for i in sel_idx]
    sel_pred = [predictions['video'][i] for i in sel_idx]

    # Set current figure
    plt.figure(args.fig.number)
    for i in range(len(sel_idx)):
        truth, obs, pred = sel_truth[i], sel_obs[i], sel_pred[i]
        m = sel_metric[i]
        # Plot start, end, and trisection points of each video
        t_max = len(obs)
        frames = [0, t_max//3, t_max-1 - t_max//3, t_max-1]
        for j, t in enumerate(frames):
            # Plot observed image
            args.axes[2*i,j].cla()
            img = obs[t].transpose((1,2,0))
            args.axes[2*i,j].imshow(img)
            args.axes[2*i,j].set_title("Observed".format(m))
            # Plot predicted image
            args.axes[2*i+1,j].cla()
            img = pred[t].transpose((1,2,0))
            args.axes[2*i+1,j].imshow(img)
            args.axes[2*i+1,j].set_title("Metric = {:0.3f}".format(m))
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
    data_dir = os.path.abspath(args.data_dir)
    all_data = WeizmannDataset(data_dir, item_as_dict=True)
    all_persons = all_data.seq_id_sets[0]
    # Leave one person out of training set
    train_data = all_data.select([['shahar'], None], invert=True)
    # Test on left out person
    test_data = all_data.select([['shahar'], None])
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
        args.modalities = ['video']

    # Load data for specified modalities
    train_data, test_data = load_data(args.modalities, args)

    # Resolve short model names to long model names
    args.model = models.names.get(args.model, args.model)

    # Construct model
    dims = {'video': 64 * 64}
    dists = {'video': 'Bernoulli'}
    if hasattr(models, args.model):
        constructor = getattr(models, args.model)
        image_encoder = models.common.ImageEncoder(z_dim=256)
        image_decoder = models.common.ImageDecoder(z_dim=256)
        model = constructor(args.modalities,
                            dims=[dims[m] for m in args.modalities],
                            dists=[dists[m] for m in args.modalities],
                            encoders={'video': image_encoder},
                            decoders={'video': image_decoder},
                            z_dim=256, h_dim=256,
                            device=args.device, **args.model_args)
        model.z0_mean.requires_grad = False
        model.z0_log_std.requires_grad = False
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
        args.fig, args.axes = plt.subplots(4, 4, figsize=(8,8),
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
                                 shuffle=False, pin_memory=False)
        with torch.no_grad():
            pred, _  = evaluate(eval_loader, model, args,
                                os.path.join(args.save_dir, "train.pdf"))
            # save_predictions(train_data, pred, pred_train_dir)
            
        print("--Testing--")
        eval_loader = DataLoader(test_data, batch_size=args.batch_size,
                                 collate_fn=mseq.seq_collate_dict,
                                 shuffle=False, pin_memory=False)
        with torch.no_grad():
            pred, _  = evaluate(eval_loader, model, args,
                                os.path.join(args.save_dir, "test.pdf"))
            # save_predictions(test_data, pred, pred_test_dir)

        # Save command line flags, model params
        save_params(args, model)
        return

    # Split training data into chunks
    if args.split is not None:
        train_data = train_data.split(args.split, n_is_len=True)
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
    parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                        help='input batch size for training (default: 50)')
    parser.add_argument('--split', type=int, default=25, metavar='K',
                        help='split data into K-sized chunks (default: 25)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--w_decay', type=float, default=0, metavar='F',
                        help='Adam weight decay (default: 0)')
    parser.add_argument('--base_rate', type=float, default=None, metavar='R',
                        help='sampling rate to resample to')
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                        help='random seed (default: 1)')
    parser.add_argument('--kld_mult', type=float, default=1.0, metavar='F',
                        help='max kld loss multiplier (default: 1.0)')
    parser.add_argument('--rec_mults', type=yaml.load, default=None,
                        help='reconstruction loss multiplier (default: 1/dim)')
    parser.add_argument('--kld_anneal', type=int, default=100, metavar='N',
                        help='epochs to increase kld_mult over (default: 100)')
    parser.add_argument('--burst_frac', type=float, default=0, metavar='F',
                        help='burst error rate during training (default: 0)')
    parser.add_argument('--drop_frac', type=float, default=0, metavar='F',
                        help='fraction of data to randomly drop at test time')
    parser.add_argument('--start_frac', type=float, default=0, metavar='F',
                        help='fraction of test trajectory to begin at')
    parser.add_argument('--stop_frac', type=float, default=1, metavar='F',
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
    parser.add_argument('--data_dir', type=str, default="./datasets/weizmann",
                        help='path to data base directory')
    parser.add_argument('--save_dir', type=str, default="./weizmann_save",
                        help='path to save models and predictions')
    parser.add_argument('--train_subdir', type=str, default='train',
                        help='training data subdirectory')
    parser.add_argument('--test_subdir', type=str, default='test',
                        help='testing data subdirectory')
    args = parser.parse_args()
    main(args)
