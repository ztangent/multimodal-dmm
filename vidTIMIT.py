"""Training code for the vidTIMIT audio-visual dataset."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from builtins import range
import os, copy
from collections import defaultdict

import numpy as np
import torch
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt

from datasets import vidTIMIT
from utils import eval_ssim
import models
import trainer

class VidTIMITTrainer(trainer.Trainer):
    """Class for training on the vidTIMIT human action dataset."""

    parser = copy.copy(trainer.Trainer.parser)

    # Rewrite split help function to be more clear
    for action in parser._actions:
        if action.dest != 'split':
            continue
        action.help = 'split each training sequence into L-sized chunks'
        action.metavar = 'L'

    # Set parameter defaults for VidTIMIT dataset
    defaults = {
        'modalities' : ['video', 'audio'],
        'batch_size' : 25, 'split' : 25, 'bylen' : True,
        'epochs' : 500, 'lr' : 5e-4,
        'rec_mults' : {'video': 1, 'audio': 1},
        'kld_anneal' : 250, 'burst_frac' : 0.1,
        'drop_frac' : 0.0, 'start_frac' : 0, 'stop_frac' : 1,
        'eval_metric' : 'rec_loss', 'viz_metric' : 'ssim',
        'eval_freq' : 10, 'save_freq' : 10,
        'data_dir' : './datasets/vidTIMIT',
        'save_dir' : './vidTIMIT_save'
    }
    parser.set_defaults(**defaults)

    def build_model(self, constructor, args):
        """Construct model using provided constructor."""
        dims = {'video': (3, 64, 64), 'audio': (10, 1281)}
        dists = {'video': 'Bernoulli', 'audio': 'Bernoulli'}
        z_dim = args.model_args.get('z_dim', 256)
        h_dim = args.model_args.get('h_dim', 256)
        gauss_out = (args.model != 'MultiDKS')
        encoders = {'video': models.common.ImageEncoder(z_dim, gauss_out),
                    'audio': models.common.AudioEncoder(z_dim, gauss_out)}
        decoders = {'video': models.common.ImageDecoder(z_dim),
                    'audio': models.common.AudioDecoder(z_dim)}
        custom_mods = [m for m in ['video', 'audio'] if m in args.modalities]
        model = constructor(args.modalities,
                            dims=[dims[m] for m in args.modalities],
                            dists=[dists[m] for m in args.modalities],
                            encoders={m: encoders[m] for m in custom_mods},
                            decoders={m: decoders[m] for m in custom_mods},
                            z_dim=z_dim, h_dim=h_dim,
                            device=args.device, **args.model_args)
        return model

    def pre_build_args(self, args):
        """Process args before model is constructed."""
        args = super(VidTIMITTrainer, self).pre_build_args(args)
        # Set up method specific model and training args
        if args.method in ['b-skip', 'f-skip', 'b-mask', 'f-mask']:
            # Use both unimodal and multimodal ELBO training loss
            args.train_args['uni_loss'] = True
        return args

    def post_build_args(self, args):
        """Process args after model is constructed."""
        # Scale up reconstruction loss depending on how much data is corrupted
        corrupt_mult = 1 / (1 - args.corrupt.get('uniform', 0.0))
        args.rec_mults = {m : args.rec_mults[m] * corrupt_mult
                          for m in args.modalities}
        return args

    def load_data(self, modalities, args):
        print("Loading data...")
        data_dir = os.path.abspath(args.data_dir)
        all_data = vidTIMIT.VidTIMITDataset(data_dir, item_as_dict=True)
        # Split into train and test set
        train_data = all_data.select([None, ['sa1', 'sa2']], invert=True)
        test_data = all_data.select([None, ['sa1', 'sa2']])
        print("Done.")
        if len(args.normalize) > 0:
            print("Normalizing ", args.normalize, "...")
            # Normalize test data using training data as reference
            test_data.normalize_(modalities=args.normalize,
                                 ref_data=train_data)
            # Normalize training data in-place
            train_data.normalize_(modalities=args.normalize)
        return train_data, test_data

    def compute_metrics(self, model, infer, prior, recon,
                        targets, mask, lengths, order, args):
        """Compute evaluation metrics from batch of inputs and outputs."""
        metrics = dict()
        t_max, b_dim = max(lengths), len(lengths)
        if type(lengths) != torch.Tensor:
            lengths = torch.FloatTensor(lengths).float().to(args.device)

        # Compute and store KLD and reconstruction losses
        metrics['kld_loss'] = model.kld_loss(infer, prior, mask).item()
        metrics['rec_loss'] = model.rec_loss(targets, recon, mask,
                                             args.rec_mults).item()

        # Compute video mean squared error and SSIM for each timestep
        rec_vid, tgt_vid = recon['video'][0], targets['video']
        v_mse = ((rec_vid - tgt_vid).pow(2) / rec_vid[0,0].numel())
        v_mse = v_mse.sum(dim=list(range(2, v_mse.dim())))
        ssim = eval_ssim(rec_vid.flatten(0, 1), tgt_vid.flatten(0, 1))
        ssim = ssim.view(t_max, b_dim)

        # Compute mask mean squared error and SSIM for each timestep
        rec_audio, tgt_audio = recon['audio'][0], targets['audio']
        a_mse = ((rec_audio - tgt_audio).pow(2) / rec_audio[0,0].numel())
        a_mse = a_mse.sum(dim=list(range(2, a_mse.dim())))

        # Average across timesteps, for each sequence
        def time_avg(val):
            val[1 - mask.squeeze(-1)] = 0.0
            return val.sum(dim = 0) / lengths
        metrics['v_mse'] = time_avg(v_mse)[order].tolist()
        metrics['ssim'] = time_avg(ssim)[order].tolist()
        metrics['a_mse'] = time_avg(a_mse)[order].tolist()

        return metrics

    def summarize_metrics(self, metrics, n_timesteps):
        """Summarize and print metrics across dataset."""
        summary = defaultdict(lambda : float('nan'))
        for key, val in list(metrics.items()):
            if type(val) is list:
                # Compute mean and std dev. of metric over sequences
                summary[key] = np.mean(val)
                summary[key + '_std'] = np.std(val)
            else:
                # Average over all timesteps
                summary[key] = val / n_timesteps
        print('Evaluation\tKLD: {:7.1f}\tRecon: {:7.1f}'.\
              format(summary['kld_loss'], summary['rec_loss']))
        print('\tVideo\tMSE: {:2.3f} +/- {:2.3f}\tSSIM: {:2.3f} +/- {:2.3f}'.\
              format(summary['v_mse'], summary['v_mse_std'],
                     summary['ssim'], summary['ssim_std']))
        print('\tAudio\tMSE: {:2.3f} +/- {:2.3f}'.\
              format(summary['a_mse'], summary['a_mse_std']))
        return summary

    def visualize(self, results, metric, args):
        """Plots predictions against truth for representative fits."""
        reference = results['targets']
        observed = results['inputs']
        predicted = results['recon']

        # Select best and worst predictions
        sel_idx = np.concatenate((np.argsort(metric)[-1:][::-1],
                                  np.argsort(metric)[:1]))
        sel_metric = [metric[i] for i in sel_idx]

        if not hasattr(args, 'fig'):
            # Create figure to visualize predictions
            args.fig, args.axes = plt.subplots(
                nrows=3*len(sel_idx), ncols=2,
                figsize=(12,4*len(sel_idx)+0.5),
                subplot_kw={'aspect': 'equal'})
        else:
            # Set current figure
            plt.figure(args.fig.number)
        axes = args.axes

        # Helper function to stitch video snapshots into storyboard
        def stitch(video, times):
            nc = video.shape[1]
            board = [np.hstack([video[t].transpose(1, 2, 0),
                                np.ones(shape=(64, 1, nc))]) for t in times]
            board = np.squeeze(np.hstack(board))
            return board

        # Helper function to plot a storyboard on current axis
        def plot_board(board, tick_labels, y_label):
            plt.cla()
            plt.xticks(np.arange(32, 65 * len(tick_labels), 65), tick_labels)
            plt.yticks([])
            if board.ndim == 2:
                plt.imshow(board, cmap='gray')
            else:
                plt.imshow(board)
            plt.ylabel(y_label)
            plt.gca().tick_params(length=0)

        # Plot video storyboards in first column
        for i in range(len(sel_idx)):
            true = reference['video'][sel_idx[i]]
            obsv = observed['video'][sel_idx[i]]
            pred = predicted['video'][sel_idx[i]][:,0]

            # Stitch equally-spaced frames into a storyboard row
            times = np.linspace(0, len(true)-1, 8, dtype=int)
            true_board = stitch(true, times)
            obsv_board = stitch(obsv, times)
            pred_board = stitch(pred, times)

            # Set missing observations to white
            obsv_board[np.isnan(obsv_board)] = 1.0

            # Remove tick labels
            labels = ['' for t in times]

            # Plot original video
            plt.sca(axes[3*i, 0])
            plot_board(true_board, labels, "Original")
            # Plot observations
            plt.sca(axes[3*i+1, 0])
            plot_board(obsv_board, labels, "Observed")
            # Plot reconstructed video
            plt.sca(axes[3*i+2, 0])
            plot_board(pred_board, labels, "Reconstructed")

            # Display metric as title on top of original video
            axes[3*i, 0].set_title('Metric: {:0.3f}'.format(sel_metric[i]),
                                   fontdict={'fontsize': 10}, loc='right')

        # Helper function to plot spectrogram on current axis
        def plot_spectrogram(audio, y_label):
            # Plot only the magnitude channels (ignore phase)
            audio = audio[:,:audio.shape[1]//2]
            # Undo overlapping of windows by picking central value
            overlap = 2
            spec = audio[:,overlap,:].T
            plt.cla()
            plt.imshow(spec, aspect='auto')
            plt.ylabel(y_label)
            plt.gca().tick_params(length=0)

        # Plot audio spectograms in second column
        for i in range(len(sel_idx)):
            true = reference['audio'][sel_idx[i]]
            obsv = observed['audio'][sel_idx[i]]
            pred = predicted['audio'][sel_idx[i]][:,0]

            # Set missing observations to white
            obsv[np.isnan(obsv)] = 1.0

            # Remove tick labels
            labels = ['' for t in times]

            # Plot original video
            plt.sca(axes[3*i, 1])
            plot_spectrogram(true, "Original")
            # Plot observations
            plt.sca(axes[3*i+1, 1])
            plot_spectrogram(obsv, "Observed")
            # Plot reconstructed video
            plt.sca(axes[3*i+2, 1])
            plot_spectrogram(pred, "Reconstructed")

            # Display metric as title on top of spectrogram
            axes[3*i, 1].set_title('Metric: {:0.3f}'.format(sel_metric[i]),
                                   fontdict={'fontsize': 10}, loc='right')

        # Remove axis borders
        for i in range(axes.shape[0]):
            for j in range(axes.shape[1]):
                for spine in axes[i,j].spines.values():
                    spine.set_visible(False)

        plt.tight_layout()
        plt.draw()
        if args.eval_set is not None:
            fig_path = os.path.join(args.save_dir, args.eval_set + '.pdf')
        else:
            fig_path = os.path.join(args.save_dir, 'visualize.pdf')
        plt.savefig(fig_path)
        plt.pause(1.0 if args.test else 0.001)

    def save_results(self, results, args):
        """Save results to video."""
        pass

if __name__ == "__main__":
    args = VidTIMITTrainer.parser.parse_args()
    trainer = VidTIMITTrainer(args)
    trainer.run(args)
