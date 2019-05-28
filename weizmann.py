"""Training code for the Weizmann human action dataset."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys, os, shutil
import argparse, yaml

import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import multiseq as mseq
from datasets import weizmann

import models
from utils import eval_ssim, anneal, plot_grad_flow

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
    # Set up accumulators
    data_num = 0
    reference = {m: [] for m in loader.dataset.modalities}
    predicted = {m: [] for m in args.modalities}
    observed = {m: [] for m in args.modalities}
    kld_loss, rec_loss, mse_loss, ssim_loss = [], [], [], []
    accuracy = {m: [] for m in ['action', 'person']}
    # Only compute reconstruction loss for specified modalities
    rec_mults = dict(args.rec_mults)
    if args.eval_mods is not None:
        for m in rec_mults:
            rec_mults[m] *= float(m in args.eval_mods)
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
        # Remove specified modalities to test conditioned generation
        for m in args.drop_mods:
            inputs[m][:] = float('nan')
        # Run forward pass using all modalities, get MAP estimate
        infer, prior, recon = model(inputs, lengths=lengths, sample=False,
                                    **args.eval_args)
        # Compute and store KLD and reconstruction losses
        kld_loss.append(model.kld_loss(infer, prior, mask))
        rec_loss.append(model.rec_loss(targets, recon, mask, args.rec_mults))
        # Keep track of total number of time-points
        data_num += sum(lengths)
        # Decollate and store observations and predictions
        for m in targets.keys():
            reference[m] += mseq.seq_decoll(targets[m], lengths, order)
        for m in recon.keys():
            observed[m] += mseq.seq_decoll(inputs[m], lengths, order)
            predicted[m] += mseq.seq_decoll(recon[m][0], lengths, order)
        # Compute video mean squared error and SSIM for each timestep
        rec_vid, tgt_vid = recon['video'][0], targets['video']
        mse = ((rec_vid - tgt_vid).pow(2) / rec_vid[0,0].numel())
        mse = mse.sum(dim=range(2, mse.dim()))
        ssim = eval_ssim(rec_vid.flatten(0, 1), tgt_vid.flatten(0, 1))
        ssim = ssim.view(max(lengths), len(lengths))
        # Average across timesteps, for each sequence
        lens = torch.tensor(lengths).float().to(args.device)
        mse[1 - mask.squeeze(-1)] = 0.0
        mse = mse.sum(dim=0) / lens
        mse_loss += mse[order].tolist()
        ssim[1 - mask.squeeze(-1)] = 0.0
        ssim = ssim.sum(dim=0) / lens
        ssim_loss += ssim[order].tolist()
        # Compute prediction accuracy for action and person labels
        for m in ['action', 'person']:
            rec, tgt = recon[m][0], targets[m]
            correct = (rec.argmax(dim=-1) == tgt.squeeze(-1).long())
            acc = correct.sum(dim=0).float() / lens
            accuracy[m] += acc[order].tolist()
    # Plot predictions against truth
    if args.visualize:
         visualize(reference, observed, predicted, ssim_loss, args, fig_path)
    # Average losses and print
    kld_loss = sum(kld_loss) / data_num
    rec_loss = sum(rec_loss) / data_num
    mse_std = np.std(mse_loss)
    mse_loss = sum(mse_loss) / len(mse_loss)
    ssim_std = np.std(ssim_loss)
    ssim_loss = sum(ssim_loss) / len(ssim_loss)
    action_std = np.std(accuracy['action'])
    action_acc = sum(accuracy['action']) / len(accuracy['action'])
    person_std = np.std(accuracy['person'])
    person_acc = sum(accuracy['person']) / len(accuracy['person'])
    losses = kld_loss, rec_loss, mse_loss, ssim_loss
    print('Evaluation\tKLD: {:7.1f}\tRecon: {:7.1f}'.\
          format(kld_loss, rec_loss))
    print('\t\tMSE: {:2.3f} +/- {:2.3f}\tSSIM: {:2.3f} +/- {:2.3f}'.\
          format(mse_loss, mse_std, ssim_loss, ssim_std))
    print('\t\tAct: {:2.3f} +/- {:2.3f}\tPers: {:2.3f} +/- {:2.3f}'.\
          format(action_acc, action_std, person_acc, person_std))
    return reference, predicted, losses

def visualize(reference, observed, predicted,
              metric, args, fig_path=None):
    """Plots predictions against truth for representative fits."""
    # Select best and worst predictions
    sel_idx = np.concatenate((np.argsort(metric)[-1:][::-1],
                              np.argsort(metric)[:1]))
    sel_metric = [metric[i] for i in sel_idx]
    sel_true = [reference['video'][i] for i in sel_idx]
    sel_obsv = [observed['video'][i] for i in sel_idx]
    sel_pred = [predicted['video'][i] for i in sel_idx]

    sel_true_act = [reference['action'][i] for i in sel_idx]
    sel_obsv_act = [observed['action'][i] for i in sel_idx]
    if 'action' in predicted.keys():
        sel_pred_act = [predicted['action'][i] for i in sel_idx]
    else:
        sel_pred_act = [None] * len(sel_idx)

    if not hasattr(args, 'fig'):
        # Create figure to visualize predictions
        args.fig, args.axes = plt.subplots(
            nrows=3*len(sel_idx), ncols=1, figsize=(8,4*len(sel_idx)+0.5),
            subplot_kw={'aspect': 'equal'})
    else:
        # Set current figure
        plt.figure(args.fig.number)
    axes = args.axes
    
    for i in range(len(sel_idx)):
        true, obsv, pred = sel_true[i], sel_obsv[i], sel_pred[i]
        t_act, o_act, p_act = sel_true_act[i], sel_obsv_act[i], sel_pred_act[i]

        # Stitch equally-spaced frames into a storyboard row
        frames = np.linspace(0, len(true)-1, 8, dtype=int)

        true_board = [np.hstack([true[t].transpose(1, 2, 0),
                                 np.ones(shape=(64, 1, 3))]) for t in frames]
        true_board = np.hstack(true_board)
        
        obsv_board = [np.hstack([obsv[t].transpose(1, 2, 0),
                                 np.ones(shape=(64, 1, 3))]) for t in frames]
        obsv_board = np.hstack(obsv_board)
        obsv_board[np.isnan(obsv_board)] = 1.0 # Set missing frames to white

        pred_board = [np.hstack([pred[t].transpose(1, 2, 0),
                                 np.ones(shape=(64, 1, 3))]) for t in frames]
        pred_board = np.hstack(pred_board)

        # Read predicted action names
        pred_probs = p_act.max(axis=1)
        p_act = [weizmann.actions[a] for a in p_act.argmax(axis=1)]
        t_labels = [weizmann.actions[int(t_act[t])] for t in frames]
        o_labels = ['' if (o_act[t] != o_act[t]) else
                    weizmann.actions[int(o_act[t])] for t in frames]
        p_labels = ['{} ({:0.1f})'.format(p_act[t], pred_probs[t])
                       for t in frames]
        
        # Plot original video
        plt.sca(axes[3*i])
        plt.cla()
        plt.xticks(np.arange(32, 65 * len(frames), 65), t_labels)
        plt.yticks([])
        plt.imshow(true_board)
        plt.ylabel("Original")
        plt.gca().tick_params(length=0)
        
        # Plot observations
        plt.sca(axes[3*i+1])
        plt.cla()
        plt.xticks(np.arange(32, 65 * len(frames), 65), o_labels)
        plt.yticks([])
        plt.imshow(obsv_board)
        plt.ylabel("Observed")
        plt.gca().tick_params(length=0)

        # Plot reconstructed video
        plt.sca(axes[3*i+2])
        plt.cla()
        plt.xticks(np.arange(32, 65 * len(frames), 65), p_labels)
        plt.yticks([])
        plt.imshow(pred_board)
        plt.ylabel("Reconstructed")
        plt.gca().tick_params(length=0)

        # Display metric as title on top of original video
        axes[3*i].set_title('Metric: {:0.3f}'.format(sel_metric[i]),
                            fontdict={'fontsize': 10}, loc='right')
        
    for i in range(len(axes)):
        for spine in axes[i].spines.values():
            spine.set_visible(False)        
        
    plt.tight_layout()
    plt.draw()
    if fig_path is not None:
        plt.savefig(fig_path)
    plt.pause(1.0 if args.test else 0.001)
    
def save_results(reference, predicted, args):
    """Save results to video."""
    print("Saving results...")
    
    for i, video in enumerate(predicted['video']):
        # Tranpose video to T * H * W * C
        video = video.transpose((0,2,3,1))
        # Construct file name as [person]_[action].avi
        p_id, a_id = reference['person'][i][0], reference['action'][i][0]
        person = weizmann.persons[int(p_id)]
        action = weizmann.actions[int(a_id)]
        path = os.path.join(args.save_dir, '{}_{}.avi'.format(person, action))
        # Create video writer for uncompressed 25 fps video
        vwriter = cv.VideoWriter(path, 0, 25.0, video.shape[1:3])

        # Iterate over frames
        for t, frame in enumerate(video):
            frame = (frame * 255).astype('uint8')
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            if 'action' in predicted:
                act_probs = predicted['action'][i][t]
                act_str = weizmann.actions[np.argmax(act_probs)]
                cv.putText(frame, act_str, (0, 15), cv.FONT_HERSHEY_SIMPLEX,
                           0.5, (255, 255, 255), 1, cv.LINE_AA)
            vwriter.write(frame)
        vwriter.release()

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
    all_data = weizmann.WeizmannDataset(data_dir, item_as_dict=True)
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
        args.modalities = ['video', 'person', 'action']

    # Load data for specified modalities
    train_data, test_data = load_data(args.modalities, args)

    # Resolve short model names to long model names
    args.model = models.names.get(args.model, args.model)

    # Construct model
    dims = {'video': (3, 64, 64), 'person': 10, 'action': 10}
    dists = {'video': 'Bernoulli',
             'person': 'Categorical',
             'action': 'Categorical'}
    if hasattr(models, args.model):
        print('Constructing model...')
        constructor = getattr(models, args.model)
        gauss_out = (args.model != 'MultiDKS')
        image_encoder = models.common.ImageEncoder(256, gauss_out)
        image_decoder = models.common.ImageDecoder(256)
        model = constructor(args.modalities,
                            dims=[dims[m] for m in args.modalities],
                            dists=[dists[m] for m in args.modalities],
                            encoders={'video': image_encoder},
                            decoders={'video': image_decoder},
                            z_dim=256, h_dim=256,
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
        args.rec_mults = {'video': 1,
                          'person': 10,
                          'action': 10}
        
    # Setup loss and optimizer
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.w_decay)

    # Create path to save models/predictions
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
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
            ref, pred, _  = evaluate(eval_loader, model, args,
                                     os.path.join(args.save_dir, "train.pdf"))
            save_results(ref, pred, args)
            
        print("--Testing--")
        eval_loader = DataLoader(test_data, batch_size=args.batch_size,
                                 collate_fn=mseq.seq_collate_dict,
                                 shuffle=False, pin_memory=False)
        with torch.no_grad():
            ref, pred, _  = evaluate(eval_loader, model, args,
                                     os.path.join(args.save_dir, "test.pdf"))
            save_results(ref, pred, args)

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
                ref, pred, losses = evaluate(test_loader, model, args)
                # Select best epoch via reconstruction loss
                _, loss, _, _ = losses
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
    parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                        help='input batch size for training (default: 50)')
    parser.add_argument('--split', type=int, default=25, metavar='K',
                        help='split data into K-sized chunks (default: 25)')
    parser.add_argument('--epochs', type=int, default=3000, metavar='N',
                        help='number of epochs to train (default: 3000)')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--w_decay', type=float, default=0, metavar='F',
                        help='Adam weight decay (default: 0)')
    parser.add_argument('--base_rate', type=float, default=None, metavar='R',
                        help='sampling rate to resample to')
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                        help='random seed (default: 1)')
    parser.add_argument('--kld_mult', type=float, default=1.0, metavar='F',
                        help='max kld loss multiplier (default: 1.0)')
    parser.add_argument('--rec_mults', type=yaml.safe_load, default=None,
                        help='reconstruction loss multiplier')
    parser.add_argument('--kld_anneal', type=int, default=1500, metavar='N',
                        help='epochs to anneal kld_mult over (default: 1500)')
    parser.add_argument('--burst_frac', type=float, default=0.2, metavar='F',
                        help='burst error rate during training (default: 0.2)')
    parser.add_argument('--drop_frac', type=float, default=0.5, metavar='F',
                        help='fraction of data to randomly drop at test time')
    parser.add_argument('--start_frac', type=float, default=0, metavar='F',
                        help='fraction of test trajectory to begin at')
    parser.add_argument('--stop_frac', type=float, default=1, metavar='F',
                        help='fraction of test trajectory to stop at')
    parser.add_argument('--drop_mods', type=str, default=[], nargs='+',
                        help='modalities to delete at test (default: none')
    parser.add_argument('--eval_mods', type=str, default=None, nargs='+',
                        help='modalities to evaluate at test (default: none')
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
