"""Training code for spirals dataset."""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from builtins import range
import os, copy

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection

from datasets.spirals import SpiralsDataset
import trainer

class SpiralsTrainer(trainer.Trainer):
    """Class for training on noisy 2D spirals."""

    parser = copy.copy(trainer.Trainer.parser)

    # Add these arguments specifically for the Spirals dataset
    parser.add_argument('--train_subdir', type=str,
                        default='train', metavar='DIR',
                        help='training data subdirectory')
    parser.add_argument('--test_subdir', type=str,
                        default='test', metavar='DIR',
                        help='testing data subdirectory')    

    # Set parameter defaults for spirals dataset
    defaults = {
        'modalities' : ['spiral-x', 'spiral-y'],
        'batch_size' : 100, 'split' : 1, 'bylen' : False,
        'epochs' : 500, 'lr' : 1e-4,
        'kld_anneal' : 100, 'burst_frac' : 0.1,
        'drop_frac' : 0.5, 'start_frac' : 0.25, 'stop_frac' : 0.75,
        'eval_metric' : 'mse', 'viz_metric' : 'mse',
        'eval_freq' : 10, 'save_freq' : 10,
        'data_dir' : './datasets/spirals',
        'save_dir' : './spirals_save'
    }
    parser.set_defaults(**defaults)
    
    def build_model(self, constructor, args):
        """Construct model using provided constructor."""
        dims = {'spiral-x': 1, 'spiral-y': 1}
        model = constructor(args.modalities,
                            dims=(dims[m] for m in args.modalities),
                            z_dim=5, h_dim=20,
                            device=args.device, **args.model_args)
        return model

    def pre_build_args(self, args):
        """Process args before model is constructed."""
        args = super(SpiralsTrainer, self).pre_build_args(args)
        # Set up method specific model and training args
        if args.method in ['b-skip', 'f-skip', 'b-mask', 'f-mask']:
            # No direct connection from features to z in encoder
            args.model_args['feat_to_z'] = False
            # Do not add unimodal ELBO training loss for RNN methods
            args.train_args['uni_loss'] = False
        return args
    
    def post_build_args(self, args):
        """Process args after model is constructed."""
        # Default reconstruction loss multipliers
        if args.rec_mults == 'auto':
            dims = self.model.dims
            corrupt_mult = 1 / (1 - args.corrupt.get('uniform', 0.0))
            args.rec_mults = {m : ((1.0 / dims[m]) / len(args.modalities)
                                   * corrupt_mult)
                              for m in args.modalities}
        return args

    def load_data(self, modalities, args):
        """Loads data for specified modalities."""
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
            test_data.normalize_(modalities=args.normalize,
                                 ref_data=train_data)
            # Normalize training data in-place
            train_data.normalize_(modalities=args.normalize)
        return train_data, test_data
    
    def compute_metrics(self, model, infer, prior, recon,
                        targets, mask, lengths, order, args):
        """Compute evaluation metrics from batch of inputs and outputs."""    
        metrics = dict()
        if type(lengths) != torch.Tensor:
            lengths = torch.FloatTensor(lengths).to(args.device)
        # Compute and store KLD and reconstruction losses
        metrics['kld_loss'] = model.kld_loss(infer, prior, mask).item()
        metrics['rec_loss'] = model.rec_loss(targets, recon, mask,
                                             args.rec_mults).item()
        # Compute mean squared error in 2D space for each time-step
        mse = sum([(recon[m][0]-targets[m]).pow(2) for m in list(recon.keys())])
        mse = mse.sum(dim=list(range(2, mse.dim())))
        # Average across timesteps, for each sequence
        def time_avg(val):
            val[1 - mask.squeeze(-1)] = 0.0
            return val.sum(dim = 0) / lengths
        metrics['mse'] = time_avg(mse)[order].tolist()    
        return metrics

    def summarize_metrics(self, metrics, n_timesteps):
        """Summarize and print metrics across dataset."""
        summary = dict()
        for key, val in list(metrics.items()):
            if type(val) is list:
                # Compute mean and std dev. of metric over sequences
                summary[key] = np.mean(val)
                summary[key + '_std'] = np.std(val)
            else:
                # Average over all timesteps
                summary[key] = val / n_timesteps
        print(('Evaluation\tKLD: {:7.1f}\tRecon: {:7.1f}\t' +
               'MSE: {:6.3f} +-{:2.3f}')\
              .format(summary['kld_loss'], summary['rec_loss'],
                      summary['mse'], summary['mse_std']))
        return summary

    def visualize(self, results, metric, args):
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
        sel_pred = [(predicted['spiral-x'][i][:,0],
                     predicted['spiral-y'][i][:,0])
                    for i in sel_idx]
        sel_rng = [(predicted['spiral-x'][i][:,1],
                    predicted['spiral-y'][i][:,1])
                    for i in sel_idx]

        # Create figure to visualize predictions
        if not hasattr(args, 'fig'):
            args.fig, args.axes = plt.subplots(4, 2, figsize=(4,8),
                                               subplot_kw={'aspect': 'equal'})
        else:
            plt.figure(args.fig.number)
        axes = args.axes

        # Set current figure
        plt.figure(args.fig.number)
        for i in range(len(sel_idx)):
            axis = args.axes[(i % 4),(i // 4)]
            # Plot spiral
            self.plot_spiral(axis, sel_true[i], sel_data[i],
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

    def plot_spiral(self, axis, true, data, obsv, pred, rng):
        """Plots a single spiral on provided axis."""
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

    def save_results(self, results, args):
        pass
        
if __name__ == "__main__":
    args = SpiralsTrainer.parser.parse_args()
    trainer = SpiralsTrainer(args)
    trainer.run(args)
