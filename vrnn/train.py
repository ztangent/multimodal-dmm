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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import seq_collate_dict, load_dataset
from models import MultiVRNN

def eval_ccc(y_true, y_pred):
    """Computes concordance correlation coefficient."""
    true_mean = np.mean(y_true)
    true_var = np.var(y_true)
    pred_mean = np.mean(y_pred)
    pred_var = np.var(y_pred)
    covar = np.cov(y_true, y_pred, bias=True)[0][1]
    ccc = 2*covar / (true_var + pred_var +  (pred_mean-true_mean) ** 2)
    return ccc

def anneal(min_val, max_val, t, anneal_len):
    if t >= anneal_len:
        return max_val
    else:
        return (max_val - min_val) * t/anneal_len

def train(loader, model, optimizer, epoch, args):
    model.train()
    loss = 0.0
    data_num = 0
    log_freq = len(loader) // args.log_freq
    # Select batches that should be supervised
    rec_mults = dict(args.rec_mults)
    for batch_num, (data, mask, lengths) in enumerate(loader):
        # Anneal KLD and supervised loss multipliers
        batch_tot = batch_num + epoch*len(loader)
        kld_mult =\
            anneal(0.0, args.kld_mult, batch_tot, args.kld_anneal*len(loader))
        sup_mult =\
            anneal(0.0, args.sup_mult, batch_tot, args.sup_anneal*len(loader))
        # Add supervised loss multiplier to reconstruction multiplier dict
        rec_mults['ratings'] = sup_mult
        # Send to device
        mask = mask.to(args.device)
        for m in data.keys():
            data[m] = data[m].to(args.device)
        # Select random samples within batch to be unsupervised
        batch_size = len(lengths)
        unsup_idx = np.random.choice(batch_size, replace=False,
                                     size=int(args.sup_ratio * batch_size))
        inputs = dict(data)
        inputs['ratings'] = torch.tensor(data['ratings'])
        inputs['ratings'][:,unsup_idx,:] = float('nan')
        # Run forward pass with all input modalities
        infer, prior, outputs = model(inputs, lengths)
        # Compute ELBO loss for all input modalities
        batch_loss = model.loss(data, infer, prior, outputs, mask,
                                kld_mult, rec_mults)
        # Compute ELBO loss for individual input modalities
        if len(args.modalities) > 1:
            for m in args.modalities:
                # Provide only modality m + ratings (which maybe be missing)
                m_inputs = {m: inputs[m], 'ratings': inputs['ratings']}
                infer, prior, outputs = model(m_inputs, lengths)
                # Compute ELBO loss for modality m and ratings
                m_inputs['ratings'] = data['ratings']
                batch_loss += model.loss(m_inputs, infer, prior, outputs, mask,
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
    print('Epoch: {}\tLoss: {:10.1f}\tB_KLD: {:0.3f}\tB_Sup: {:5.2f}'.\
          format(epoch, loss, kld_mult, sup_mult))
    return loss

def evaluate(dataset, model, args, fig_path=None):
    model.eval()
    predictions = []
    data_num = 0
    kld_loss, rec_loss, sup_loss = 0.0, 0.0, 0.0
    corr, ccc = [], []
    for data, orig in zip(dataset, dataset.orig['ratings']):
        # Collate data into batch dictionary of size 1
        data, mask, lengths = seq_collate_dict([data])
        # Send to device
        mask = mask.to(args.device)
        for m in data.keys():
            data[m] = data[m].to(args.device)
        # Separate target modality from input modalities
        target = {'ratings': data['ratings']}
        inputs = dict(data)
        del inputs['ratings']
        # Run forward pass using input modalities
        infer, prior, outputs = model(inputs, lengths)
        # Compute and store KLD, reconstruction and supervised losses
        kld_loss += model.kld_loss(infer, prior, mask)
        rec_loss += model.rec_loss(inputs, outputs, mask, args.rec_mults)
        sup_loss += model.rec_loss(target, outputs, mask)
        # Keep track of total number of time-points
        data_num += sum(lengths)
        # Resize predictions to match original length
        out_mean, _ = outputs
        pred = out_mean['ratings'][:lengths[0], 0].view(-1).cpu().numpy()
        pred = np.repeat(pred, int(dataset.ratios['ratings']))[:len(orig)]
        if len(pred) < len(orig):
            pred = np.concatenate((pred, pred[len(pred)-len(orig):]))
        predictions.append(pred)
        # Compute correlation and CCC of predictions against ratings
        corr.append(pearsonr(orig.reshape(-1), pred)[0])
        ccc.append(eval_ccc(orig.reshape(-1), pred))
    # Plot predictions against ratings
    if args.visualize:
        plot_predictions(dataset, predictions, ccc, args, fig_path)
    # Average losses and print
    kld_loss /= data_num
    rec_loss /= data_num
    sup_loss /= data_num
    losses = kld_loss, rec_loss, sup_loss
    print('Evaluation\tKLD: {:7.1f}\tRecon: {:7.1f}\tSup: {:7.1f}'.\
          format(*losses))
    # Average statistics and print
    stats = {'corr': np.mean(corr), 'corr_std': np.std(corr),
             'ccc': np.mean(ccc), 'ccc_std': np.std(ccc)}
    print('Corr: {:0.3f}\tCCC: {:0.3f}'.format(stats['corr'], stats['ccc']))
    return predictions, losses, stats

def plot_predictions(dataset, predictions, metric, args, fig_path=None):
    """Plots predictions against ratings for representative fits."""
    # Select top 4 and bottom 4
    sel_idx = np.concatenate((np.argsort(metric)[-4:][::-1],
                              np.argsort(metric)[:4]))
    sel_metric = [metric[i] for i in sel_idx]
    sel_true = [dataset.orig['ratings'][i] for i in sel_idx]
    sel_pred = [predictions[i] for i in sel_idx]
    for i, (true, pred, m) in enumerate(zip(sel_true, sel_pred, sel_metric)):
        j, i = (i // 4), (i % 4)
        args.axes[i,j].cla()
        args.axes[i,j].plot(true, 'b-')
        args.axes[i,j].plot(pred, 'c-')
        args.axes[i,j].set_xlim(0, len(true))
        args.axes[i,j].set_ylim(-1, 1)
        args.axes[i,j].set_title("Fit = {:0.3f}".format(m))
    plt.tight_layout()
    plt.draw()
    if fig_path is not None:
        plt.savefig(fig_path)
    plt.pause(1.0 if args.test else 0.001)

def save_predictions(dataset, predictions, path):
    for p, seq_id in zip(predictions, dataset.seq_ids):
        df = pd.DataFrame(p, columns=['rating'])
        fname = "target_{}_{}_normal.csv".format(*seq_id)
        df.to_csv(os.path.join(path, fname), index=False)

def save_params(args, model, train_stats, test_stats):
    fname = 'param_hist.tsv'
    df = pd.DataFrame([vars(args)], columns=vars(args).keys())
    df = df[['save_dir', 'modalities', 'normalize', 'batch_size', 'split',
             'epochs', 'lr', 'kld_mult', 'sup_mult', 'rec_mults',
             'kld_anneal', 'sup_anneal', 'sup_ratio', 'base_rate']]
    for k in ['ccc_std', 'ccc']:
        v = train_stats.get(k, float('nan'))
        df.insert(0, 'train_' + k, v)
    for k in ['ccc_std', 'ccc']:
        v = test_stats.get(k, float('nan'))
        df.insert(0, 'test_' + k, v)
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

def load_data(modalities, data_dir, normalize=[]):
    print("Loading data...")
    train_data = load_dataset(modalities, data_dir, 'Train',
                              base_rate=args.base_rate,
                              truncate=True, item_as_dict=True)
    test_data = load_dataset(modalities, data_dir, 'Valid',
                             base_rate=args.base_rate,
                             truncate=True, item_as_dict=True)
    print("Done.")
    if len(normalize) > 0:
        print("Normalizing ", normalize, "...")
        # Normalize test data using training data as reference
        test_data.normalize_(modalities=normalize, ref_data=train_data)
        # Normailze training data in-place
        train_data.normalize_(modalities=normalize)
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
        # Default to acoustic if unspecified
        args.modalities = ['acoustic', 'linguistic', 'emotient']

    # Load data for specified modalities
    train_data, test_data = load_data(args.modalities, args.data_dir,
                                      args.normalize)
    
    # Construct multimodal LSTM model
    dims = {'acoustic': 988, 'linguistic': 300,
            'emotient': 20, 'ratings': 1}
    model = MultiVRNN(args.modalities + ['ratings'],
                      dims=(dims[m] for m in (args.modalities + ['ratings'])),
                      device=args.device)
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
            pred, _, train_stats = evaluate(train_data, model, args,
                os.path.join(args.save_dir, "train.png"))
            save_predictions(train_data, pred, pred_train_dir)
            print("--Testing--")
            pred, _, test_stats = evaluate(test_data, model, args,
                os.path.join(args.save_dir, "test.png"))
            save_predictions(test_data, pred, pred_test_dir)
        # Save command line flags, model params and CCC value
        save_params(args, model, train_stats, test_stats)
        return train_stats['ccc'], test_stats['ccc']

    # Split training data into chunks
    train_data = train_data.split(args.split)
    # Batch data using data loaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=True, collate_fn=seq_collate_dict,
                              pin_memory=True)
   
    # Train and save best model
    best_ccc = -1
    best_stats = dict()
    for epoch in range(1, args.epochs + 1):
        print('---')
        train(train_loader, model, optimizer, epoch, args)
        if epoch % args.eval_freq == 0:
            with torch.no_grad():
                pred, loss, stats =\
                    evaluate(test_data, model, args)
            if stats['ccc'] > best_ccc:
                best_ccc = stats['ccc']
                best_stats = stats
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
    save_params(args, model, dict(), best_stats)
    
    return best_ccc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--modalities', type=str, default=None, nargs='+',
                        help='input modalities (default: all')
    parser.add_argument('--batch_size', type=int, default=25, metavar='N',
                        help='input batch size for training (default: 25)')
    parser.add_argument('--split', type=int, default=1, metavar='N',
                        help='sections to split each video into (default: 1)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: 1e-5)')
    parser.add_argument('--sup_ratio', type=float, default=0.5, metavar='F',
                        help='teacher-forcing ratio (default: 0.5)')
    parser.add_argument('--base_rate', type=float, default=2.0, metavar='N',
                        help='sampling rate to resample to (default: 2.0)')
    parser.add_argument('--kld_mult', type=float, default=1.0, metavar='F',
                        help='max kld loss multiplier (default: 1.0)')
    parser.add_argument('--sup_mult', type=float, default=10, metavar='F',
                        help='max supervised loss multiplier (default: 10)')
    parser.add_argument('--rec_mults', type=float, default=None, nargs='+',
                        help='reconstruction loss multiplier (default: 1/dims')
    parser.add_argument('--kld_anneal', type=int, default=100, metavar='N',
                        help='epochs to increase kld_mult over (default: 100)')
    parser.add_argument('--sup_anneal', type=int, default=100, metavar='N',
                        help='epochs to increase sup_mult over (default: 100)')
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
    parser.add_argument('--data_dir', type=str, default="../../data",
                        help='path to data base directory')
    parser.add_argument('--save_dir', type=str, default="./lstm_save",
                        help='path to save models and predictions')
    args = parser.parse_args()
    main(args)
