"""Learning with partial/missing data for spirals dataset."""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os, argparse, yaml

import ray
import ray.tune as tune

from spirals import SpiralsTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)
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
    parser.add_argument('--exp_name', type=str, default="spirals_partial",
                        help='experiment name')
    parser.add_argument('--config', type=yaml.safe_load, default={},
                        help='trial configuration arguments')
    args = parser.parse_args()

    # If max resources not specified, default to maximum - 1
    if args.max_cpus is None:
        import psutil
        args.max_cpus = min(1, psutil.cpu_count() - 1)
    if args.max_gpus is None:
        import torch
        args.max_gpus = min(1, torch.cuda.device_count() - 1)
    
    ray.init(num_cpus=args.max_cpus, num_gpus=args.max_gpus)

    trainable = lambda c, r : SpiralsTrainer.tune(c, r)
    tune.register_trainable("spirals_tune", trainable)

    # Convert data dir to absolute path so that Ray trials can find it
    data_dir = os.path.abspath(SpiralsTrainer.defaults['data_dir'])

    # Set up trial configuration
    config = {
        "data_dir": data_dir,
        "eval_args": {'flt_particles': 200},
        # Set low learning rate to prevent NaNs
        "lr": 5e-3,
        # Repeat each configuration with different random seeds
        "seed": tune.grid_search(range(args.n_repeats)),
        # Iterate over uniform data deletion in 10% steps
        "corrupt": tune.grid_search([{'uniform': i/10} for i in range(10)])
    }
    # Update config with parameters from command line
    config.update(args.config)
    
    # Run trials
    trials = tune.run(
        "spirals_tune",
        name=args.exp_name,
        config=config,
        local_dir=args.local_dir,
        resources_per_trial={"cpu": args.trial_cpus, "gpu": args.trial_gpus}
    )
