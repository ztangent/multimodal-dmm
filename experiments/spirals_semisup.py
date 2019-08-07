"""Semi-supervised learning experiments for the spirals dataset."""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os, argparse, yaml

import pandas as pd
import ray
import ray.tune as tune

from spirals import SpiralsTrainer
from .analysis import ExperimentAnalysis

parser = argparse.ArgumentParser(formatter_class=
                                 argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--analyze', action='store_true', default=False,
                    help='analyze without running experiments')
parser.add_argument('--n_repeats', type=int, default=10, metavar='N',
                    help='number of repetitions per config set')
parser.add_argument('--trial_cpus', type=int, default=1, metavar='N',
                    help='number of CPUs per trial')
parser.add_argument('--trial_gpus', type=int, default=0, metavar='N',
                    help='number of GPUs per trial')
parser.add_argument('--max_cpus', type=int, default=None, metavar='N',
                    help='max CPUs for all trials')
parser.add_argument('--max_gpus', type=int, default=None, metavar='N',
                    help='max GPUs for all trials')
parser.add_argument('--local_dir', type=str, default="./",
                help='path to Ray results')
parser.add_argument('--exp_name', type=str, default="spirals_semisup",
                    help='experiment name')
parser.add_argument('--config', type=yaml.safe_load, default={},
                    help='trial configuration arguments')
parser.add_argument('--method', type=str, default='bfvi', metavar='S',
                    help='inference method: bfvi, b/f-mask, or b/f-skip')

def run(args):
    """Runs Ray experiments."""
    # If max resources not specified, default to maximum - 1
    if args.max_cpus is None:
        import psutil
        args.max_cpus = min(1, psutil.cpu_count() - 1)
    if args.max_gpus is None:
        import torch
        args.max_gpus = min(1, torch.cuda.device_count() - 1)
    
    ray.init(num_cpus=args.max_cpus, num_gpus=args.max_gpus)

    # Convert data dir to absolute path so that Ray trials can find it
    data_dir = os.path.abspath(SpiralsTrainer.defaults['data_dir'])
            
    # Set up trial configuration
    config = {
        "data_dir": data_dir,
        # Set low learning rate to prevent NaNs
        "lr": 5e-3,
        # Repeat each configuration with different random seeds
        "seed": tune.grid_search(range(args.n_repeats)),
        # Delete spiral y-coordinates in 10% steps
        "corrupt": tune.grid_search([{'semi': i/10, 'modalities': ['spiral-y']}
                                     for i in range(10)])
    }

    # Set up model and eval args
    if args.method not in ['bfvi', 'f-mask', 'b-mask', 'f-skip', 'b-skip']:
        args.method = 'bfvi'
    if args.method == 'bfvi':
        config['model'] = 'dmm'
        config['eval_args'] = {'flt_particles': 200}
    else:
        config['model'] = 'dks'
        config['model_args'] = {
            "rnn_skip" : 'skip' in args.method,
            "rnn_dir" : 'bwd' if args.method[0] == 'b' else 'fwd',
            "feat_to_z" : False
        }
        config['train_args'] = {'uni_loss': False}
    
    # Update config with parameters from command line
    config.update(args.config)
    
    # Register trainable and run trials
    trainable = lambda c, r : SpiralsTrainer.tune(c, r)
    tune.register_trainable("spirals_tune", trainable)

    trials = tune.run(
        "spirals_tune",
        name=args.exp_name,
        config=config,
        local_dir=args.local_dir,
        resources_per_trial={"cpu": args.trial_cpus, "gpu": args.trial_gpus}
    )

def analyze(args):
    """Analyzes experiment results in log directory."""
    exp_dir = os.path.join(args.local_dir, args.exp_name)
    ea = ExperimentAnalysis(exp_dir)
    df = ea.dataframe().sort_values(['trial_id'])
    losses = dict()
    
    # Iterate across trials
    for i, trial in df.iterrows():
        print("Trial:", trial['experiment_tag'])
        try:
            trial_df = ea.trial_dataframe(trial['trial_id'])
        except(ValueError, pd.errors.EmptyDataError):
            print("No progress data to read for trial, skipping...")
            continue
        del_frac = trial['config:corrupt:semi']
        best_loss = trial_df.mean_loss.min()
        print("Best loss:", best_loss)
        print("---")

        # Store best loss for each trial and each level of deletion
        if del_frac not in losses:
            losses[del_frac] = []
        losses[del_frac].append(best_loss)

    # Print average of the best 3 losses per deletion fraction
    print("del_frac\tloss")
    del_fracs = sorted(losses.keys())
    for del_frac in del_fracs:
        best_losses = sorted(losses[del_frac])[:1]
        losses[del_frac] = sum(best_losses) / 1
        print("{}\t\t{:0.3f}".format(del_frac, losses[del_frac]))

    # Save losses to CSV file
    losses_df = pd.DataFrame.from_dict(losses, orient='index',
                                       columns=['loss']).sort_index()
    losses_df.index.name = 'del_frac'
    losses_df.to_csv(os.path.join(exp_dir, 'best_losses.csv'))
        
if __name__ == "__main__":
    args = parser.parse_args()
    if not args.analyze:
        run(args)
    analyze(args)
