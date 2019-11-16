"""Train and compare methods on a suite of inference tasks (Spirals)."""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from builtins import range
import os, argparse, yaml
import copy

import pandas as pd
import ray
import ray.tune as tune

from spirals import SpiralsTrainer
from .analysis import ExperimentAnalysis

parser = argparse.ArgumentParser(formatter_class=
                                 argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--analyze', action='store_true', default=False,
                    help='analyze without running experiments')
parser.add_argument('--n_repeats', type=int, default=1, metavar='N',
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
parser.add_argument('--exp_name', type=str, default="spirals_suite",
                    help='experiment name')
parser.add_argument('--config', type=yaml.safe_load, default={},
                    help='trial configuration arguments')

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
        "seed": tune.grid_search(list(range(args.n_repeats))),
        # Iterate across inference methods
        "method": tune.grid_search(['bfvi', 'b-mask', 'f-mask',
                                    'b-skip', 'f-skip'])
    }

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

    metrics = ['mean_loss', 'mse']
    run_results = {m: [] for m in metrics}
    run_results['method'] = []

    tasks = ['recon', 'half', 'fwd', 'bwd', 'condgen']
    task_results = {task: [] for task in tasks}
    task_results_std = {task: [] for task in tasks}
    task_results['method'] = []
    task_results_std['method'] = []

    # Iterate across trials
    for i, trial in df.iterrows():
        print('===')
        print("Trial:", trial['experiment_tag'])
        print('===')
        try:
            trial_df = ea.trial_dataframe(trial['trial_id'])
        except(ValueError, pd.errors.EmptyDataError):
            print("No progress data to read for trial, skipping...")
            continue
        method = trial['method']
        # Find index of best loss for trial
        best_idx = trial_df.mean_loss.idxmin()
        trial_results = {m: trial_df[m].iloc[best_idx] for m in metrics}
        print("Best loss:", trial_results['mean_loss'])
        print("Best MSE:", trial_results['mse'])
        print("---")

        # Store best results for each trial
        run_results['method'].append(method)
        for m in metrics:
            run_results[m].append(trial_results[m])

        # Get trial config and directory
        trial_config = ea._checkpoints[i]['config']
        trial_dir = os.path.basename(trial['logdir'])
        trial_dir = os.path.join(exp_dir, trial_dir)

        # Run evaluation suite on best saved model
        _, _, task_metrics, task_std = evaluate(trial_config, trial_dir)
        task_results['method'].append(method)
        task_results_std['method'].append(method)
        for task in tasks:
            task_results[task].append(task_metrics[task])
            task_results_std[task].append(task_std[task])

    # Print run results for each method
    run_results = pd.DataFrame(run_results)
    run_results = run_results.groupby('method').mean()
    print(run_results)

    # Print task results for each method
    task_results = pd.DataFrame(task_results)
    task_results = task_results.groupby('method').mean()
    print(task_results)

    # Print task standard deviations for each method
    task_results_std = pd.DataFrame(task_results_std)
    task_results_std = task_results_std.groupby('method').mean()
    print(task_results_std)

    # Save results to CSV file
    run_results.to_csv(os.path.join(exp_dir, 'run_results.csv'))
    task_results.to_csv(os.path.join(exp_dir, 'task_results.csv'))
    task_results_std.to_csv(os.path.join(exp_dir, 'task_results_std.csv'))

def evaluate(trial_config, trial_dir):
    """Evaluate best saved model for trial on suite of inference tasks."""
    # Inference task names
    tasks = ['recon', 'half', 'fwd', 'bwd', 'condgen']
    # Evaluation arguments for inference tasks
    task_args = {
        # Full reconstruction
        'recon': {'drop_frac': 0.0, 'start_frac': 0.0, 'stop_frac': 1.0},
        # Reconstruction after dropping half
        'half': {'drop_frac': 0.5, 'start_frac': 0.0, 'stop_frac': 1.0},
        # Forward extrapolation of last 25%
        'fwd': {'drop_frac': 0.0, 'start_frac': 0.0, 'stop_frac': 0.75},
        # Backward extrapolation of first 25%
        'bwd': {'drop_frac': 0.0, 'start_frac': 0.25, 'stop_frac': 1.0},
        # Conditional generation of last 75% of y-coordinates
        'condgen': {'drop_frac': 0.0, 'start_frac': 0.0, 'stop_frac': 0.25,
                    'keep_mods': ['spiral-x']},
    }
    # Relevant metrics for each inference task
    task_metric_names = {task: 'mse' for task in tasks}

    # Set up default args
    base_args = SpiralsTrainer.parser.parse_args([])
    # Override trainer args with trial config
    vars(base_args).update(trial_config)
    # Set load path to best model in original save dir
    base_args.load = os.path.join(trial_dir, base_args.save_dir, 'best.pth')

    # Iterate across inference tasks
    task_train_metrics, task_train_std = {}, {}
    task_test_metrics, task_test_std = {}, {}
    for task in tasks:
        print("==Inference Task: '{}'==".format(task))
        args = copy.deepcopy(base_args)
        vars(args).update(task_args[task])
        # Name save directory after task
        args.save_dir = os.path.join(trial_dir, task + '_save')
        # Construct trainer and evaluate
        trainer = SpiralsTrainer(args)
        train_metrics, test_metrics = trainer.run_eval(args)
        # Save relevant metric for task
        metric_name = task_metric_names[task]
        task_train_metrics[task] = train_metrics[metric_name]
        task_test_metrics[task] = test_metrics[metric_name]
        task_train_std[task] = train_metrics[metric_name + '_std']
        task_test_std[task] = test_metrics[metric_name + '_std']

    return (task_train_metrics, task_train_std,
            task_test_metrics, task_test_std)

if __name__ == "__main__":
    args = parser.parse_args()
    if not args.analyze:
        run(args)
    analyze(args)
