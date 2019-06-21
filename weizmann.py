"""Training code for the Weizmann human action dataset."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os, copy

import numpy as np
import torch
import cv2 as cv
import matplotlib.pyplot as plt

from datasets import weizmann
from utils import eval_ssim
import models
import trainer

class WeizmannTrainer(trainer.Trainer):
    """Class for training on the Weizmann human action dataset."""

    parser = copy.copy(trainer.Trainer.parser)

    # Rewrite split help function to be more clear
    for action in parser._actions:
        if action.dest != 'split':
            continue
        action.help = 'split each training sequence into L-sized chunks'
        action.metavar = 'L'
    
    # Set parameter defaults for Weizmann dataset
    defaults = {
        'modalities' : ['video', 'person', 'action'],
        'batch_size' : 50, 'split' : 25, 'bylen' : True,
        'epochs' : 3000, 'lr' : 5e-4,
        'rec_mults' : {'video': 1, 'person': 10, 'action': 10},
        'kld_anneal' : 1500, 'burst_frac' : 0.2,
        'drop_frac' : 0.5, 'start_frac' : 0, 'stop_frac' : 1,
        'eval_metric' : 'rec_loss', 'viz_metric' : 'ssim',
        'eval_freq' : 10, 'save_freq' : 10,
        'data_dir' : './datasets/weizmann',
        'save_dir' : './weizmann_save'
    }
    parser.set_defaults(**defaults)
    
    def build_model(self, constructor, args):
        """Construct model using provided constructor."""
        dims = {'video': (3, 64, 64), 'person': 10, 'action': 10}
        dists = {'video': 'Bernoulli',
                 'person': 'Categorical',
                 'action': 'Categorical'}
        z_dim = args.model_args.get('z_dim', 256)
        h_dim = args.model_args.get('h_dim', 256)
        gauss_out = (args.model != 'MultiDKS')
        image_encoder = models.common.ImageEncoder(z_dim, gauss_out)
        image_decoder = models.common.ImageDecoder(z_dim)
        model = constructor(args.modalities,
                            dims=[dims[m] for m in args.modalities],
                            dists=[dists[m] for m in args.modalities],
                            encoders={'video': image_encoder},
                            decoders={'video': image_decoder},
                            z_dim=z_dim, h_dim=h_dim,
                            device=args.device, **args.model_args)
        return model

    def default_args(self, args):
        """Fill unspecified args with default values."""
        return args

    def load_data(self, modalities, args):
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
        if type(lengths) != torch.tensor:
            lengths = torch.tensor(lengths).float().to(args.device)

        # Compute and store KLD and reconstruction losses
        metrics['kld_loss'] = model.kld_loss(infer, prior, mask)
        metrics['rec_loss'] = model.rec_loss(targets, recon, mask,
                                             args.rec_mults)

        # Compute video mean squared error and SSIM for each timestep
        rec_vid, tgt_vid = recon['video'][0], targets['video']
        mse = ((rec_vid - tgt_vid).pow(2) / rec_vid[0,0].numel())
        mse = mse.sum(dim=range(2, mse.dim()))
        ssim = eval_ssim(rec_vid.flatten(0, 1), tgt_vid.flatten(0, 1))
        ssim = ssim.view(t_max, b_dim)

        # Average across timesteps, for each sequence
        def time_avg(val):
            val[1 - mask.squeeze(-1)] = 0.0
            return val.sum(dim = 0) / lengths
        metrics['mse'] = time_avg(mse)[order].tolist()
        metrics['ssim'] = time_avg(ssim)[order].tolist()

        # Compute prediction accuracy over time for action and person labels
        def time_acc(probs, targets):
            correct = (probs.argmax(dim=-1) == targets.squeeze(-1).long())
            return correct.sum(dim=0).float() / lengths
        for m in ['action', 'person']:
            if m not in recon:
                metrics[m] = [0] * b_dim
                continue
            metrics[m] = time_acc(recon[m][0], targets[m])
            metrics[m] = metrics[m][order].tolist()

        return metrics

    def summarize_metrics(self, metrics, n_timesteps):
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
        print('Evaluation\tKLD: {:7.1f}\tRecon: {:7.1f}'.\
              format(summary['kld_loss'], summary['rec_loss']))
        print('\t\tMSE: {:2.3f} +/- {:2.3f}\tSSIM: {:2.3f} +/- {:2.3f}'.\
              format(summary['mse'], summary['mse_std'],
                     summary['ssim'], summary['ssim_std']))
        print('\t\tAct: {:2.3f} +/- {:2.3f}\tPers: {:2.3f} +/- {:2.3f}'.\
              format(summary['action'], summary['action_std'],
                     summary['person'], summary['person_std']))
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
        sel_true = [reference['video'][i] for i in sel_idx]
        sel_obsv = [observed['video'][i] for i in sel_idx]
        sel_pred = [predicted['video'][i][:,0] for i in sel_idx]

        sel_true_act = [reference['action'][i] for i in sel_idx]
        sel_obsv_act = [observed['action'][i] for i in sel_idx]
        if 'action' in predicted.keys():
            sel_pred_act = [predicted['action'][i][:,0] for i in sel_idx]
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

        # Helper function to stitch video snapshots into storyboard
        def stitch(video, times):
            board = [np.hstack([video[t].transpose(1, 2, 0),
                                np.ones(shape=(64, 1, 3))]) for t in times]
            return np.hstack(board)

        # Helper function to plot a storyboard on current axis
        def plot_board(board, tick_labels, y_label):
            plt.cla()
            plt.xticks(np.arange(32, 65 * len(tick_labels), 65), tick_labels)
            plt.yticks([])
            plt.imshow(board)
            plt.ylabel(y_label)
            plt.gca().tick_params(length=0)

        for i in range(len(sel_idx)):
            true, obsv, pred = sel_true[i], sel_obsv[i], sel_pred[i]
            t_act, o_act, p_act =\
                (sel_true_act[i], sel_obsv_act[i], sel_pred_act[i])

            # Stitch equally-spaced frames into a storyboard row
            times = np.linspace(0, len(true)-1, 8, dtype=int)
            true_board = stitch(true, times)
            obsv_board = stitch(obsv, times)
            pred_board = stitch(pred, times)

            # Set missing observations to white
            obsv_board[np.isnan(obsv_board)] = 1.0 

            # Read predicted action names
            pred_probs = p_act.max(axis=1)
            p_act = [weizmann.actions[a] for a in p_act.argmax(axis=1)]
            t_labels = [weizmann.actions[int(t_act[t])] for t in times]
            o_labels = ['' if (o_act[t] != o_act[t]) else
                        weizmann.actions[int(o_act[t])] for t in times]
            p_labels = ['{} ({:0.1f})'.format(p_act[t], pred_probs[t])
                        for t in times]

            # Plot original video
            plt.sca(axes[3*i])
            plot_board(true_board, t_labels, "Original")
            # Plot observations
            plt.sca(axes[3*i+1])
            plot_board(obsv_board, o_labels, "Observed")
            # Plot reconstructed video
            plt.sca(axes[3*i+2])
            plot_board(pred_board, p_labels, "Reconstructed")

            # Display metric as title on top of original video
            axes[3*i].set_title('Metric: {:0.3f}'.format(sel_metric[i]),
                                fontdict={'fontsize': 10}, loc='right')

        # Remove axis borders
        for i in range(len(axes)):
            for spine in axes[i].spines.values():
                spine.set_visible(False)        

        plt.tight_layout()
        plt.draw()
        if args.eval_set is not None:
            fig_path = os.path.join(args.save_dir, args.eval_set + '.pdf')
            plt.savefig(fig_path)
        plt.pause(1.0 if args.test else 0.001)

    def save_results(self, results, args):
        """Save results to video."""
        print("Saving results...")
        reference = results['targets']
        observed = results['inputs']
        predicted = results['recon']

        # Default save args
        save_args = {'one_file': True,
                     'filename': args.eval_set + '.avi',
                     'labels': True,
                     'comparison': True}
        save_args.update(args.save_args)

        # Define frame rate and video dimensions
        shape = reference['video'][0].shape[2:4]
        if save_args['comparison']:
            shape = (shape[0]*3, shape[1])
        fps = 25.0

        # Create video writer for single output file
        if save_args['one_file']:
            path = os.path.join(args.save_dir, save_args['filename'])
            vwriter = cv.VideoWriter(path, 0, fps, shape)

        # Helper functions
        def preprocess(frame):
            return cv.cvtColor((frame * 255).astype('uint8'), cv.COLOR_RGB2BGR)
        def add_label(image, text, pos):
            cv.putText(image, text, pos, cv.FONT_HERSHEY_SIMPLEX,
                       0.4, (255, 255, 255), 1, cv.LINE_AA)

        # Iterate over videos
        for i in range(len(reference['video'])):
            # Transpose videos to T * H * W * C
            r_vid = reference['video'][i].transpose((0,2,3,1))
            o_vid = observed['video'][i].transpose((0,2,3,1))
            p_vid = predicted['video'][i][:,0].transpose((0,2,3,1))

            if not save_args['one_file']:
                # Construct file name as [person]_[action].avi
                p_id, a_id =\
                    (reference['person'][i][0], reference['action'][i][0])
                person = weizmann.persons[int(p_id)]
                action = weizmann.actions[int(a_id)]
                path = '{}_{}.avi'.format(person, action)
                path = os.path.join(args.save_dir, path)
                # Create video writer for file
                vwriter = cv.VideoWriter(path, 0, fps, shape)

            # Iterate over frames
            for t in range(len(p_vid)):
                frame = preprocess(p_vid[t])
                if save_args['labels']:
                    # Add text labels
                    if 'action' in predicted:
                        probs = predicted['action'][i][t,0]
                        text = weizmann.actions[np.argmax(probs)]
                        add_label(frame, text, (2, 10))
                    if 'person' in predicted:
                        probs = predicted['person'][i][t,0]
                        text = weizmann.persons[np.argmax(probs)]
                        add_label(frame, text, (2, 60))

                if not save_args['comparison']:
                    vwriter.write(frame)
                    continue

                # Combine frames for side-by-side comparison
                p_frame = frame
                r_frame, o_frame = preprocess(r_vid[t]), preprocess(o_vid[t])
                if save_args['labels']:
                    # Add text labels
                    r_idx = reference['action'][i][t]
                    o_idx = observed['action'][i][t]
                    r_text = weizmann.actions[int(r_idx)]
                    add_label(r_frame, r_text, (2, 10))
                    if o_idx == o_idx: #NaN check
                        o_text = weizmann.actions[int(o_idx)]
                        add_label(o_frame, o_text, (2, 10))

                    r_idx = reference['person'][i][t]
                    o_idx = observed['person'][i][t]
                    r_text = weizmann.persons[int(r_idx)]
                    add_label(r_frame, r_text, (2, 60))
                    if o_idx == o_idx: #NaN check
                        o_text = weizmann.persons[int(o_idx)]
                        add_label(o_frame, o_text, (2, 60))
                frame = np.hstack([r_frame, o_frame, p_frame])
                vwriter.write(frame)

            if not save_args['one_file']:
                vwriter.release()

        if save_args['one_file']:
            vwriter.release()
    
if __name__ == "__main__":
    args = WeizmannTrainer.parser.parse_args()
    trainer = WeizmannTrainer(args)
    trainer.run(args)
