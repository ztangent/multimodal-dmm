from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import argparse

import numpy as np
import numpy.random as r
import pandas as pd

if __name__ == '__main__':
    from multiseq import MultiseqDataset, seq_collate
else:
    from .multiseq import MultiseqDataset, seq_collate

class SpiralsDataset(MultiseqDataset):
    """Dataset of noisy spirals."""

    def __init__(self, modalities, base_dir, subset,
                 base_rate=None, truncate=False, item_as_dict=False):
        # Generate dataset if it doesn't exist yet
        subset_dir = os.path.join(base_dir, subset)
        if not os.path.exists(subset_dir):
            gen_dataset(data_dir=base_dir)
        # Load x and y as separate modalities
        dirs = {
            'spiral-x': subset_dir,
            'spiral-y': subset_dir,
        }
        regex = {
            'spiral-x': "spiral_(\d+)\.csv",
            'spiral-y': "spiral_(\d+)\.csv"
        }
        rates = {'spiral-x': 1, 'spiral-y': 1}
        preprocess = {
            # Keep only noisy x coordinates
            'spiral-x': lambda df : df.loc[:,['noisy_x']],
            # Keep only noisy y coordinates
            'spiral-y': lambda df : df.loc[:,['noisy_y']]
        }
        super(SpiralsDataset, self).__init__(
            modalities,
            [dirs[m] for m in modalities],
            [regex[m] for m in modalities],
            [preprocess[m] for m in modalities],
            [rates[m] for m in modalities],
            base_rate, truncate, item_as_dict)

def gen_spiral(start_r, stop_r, start_theta, stop_theta,
               aspect_ratio=1, timesteps=100):
    r = np.linspace(start_r, stop_r, timesteps)
    theta = np.linspace(start_theta, stop_theta, timesteps)
    x = aspect_ratio * r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def gen_dataset(n_examples=1000, n_train=600,
                timesteps=100, data_dir='./spirals'):
    # Create output dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        os.makedirs(os.path.join(data_dir, 'train'))
        os.makedirs(os.path.join(data_dir, 'test'))
    # Reset random seed for consistency    
    r.seed(1)
    # Shuffle indices
    indices = list(range(n_examples))
    r.shuffle(indices)
    # Generate spirals
    spirals = []
    for i in range(n_examples):
        # First half are CW spirals, second half are CCW spirals
        direction = 1 if (i >= n_examples/2) else -1
        # Sample start and stop radiuses
        start_r = 0.25 + r.random() * 0.5
        stop_r = 2.25 + r.random() * 0.5
        # Sample start and stop angles
        start_theta = direction * (r.random() * np.pi)
        stop_theta = direction * (r.random() * np.pi + np.pi*4)
        # Sample aspect ratio using logarithmic prior
        aspect_ratio = 2 ** (2*r.random()-1)
        # Generate spiral x-y coordinates
        x, y = gen_spiral(start_r, stop_r, start_theta, stop_theta,
                          aspect_ratio, timesteps)
        # Add Gaussian noise
        noisy_x = x + 0.1 * r.randn(timesteps)
        noisy_y = y + 0.1 * r.randn(timesteps)
        spiral = np.stack([x, y, noisy_x, noisy_y], axis=1)
        spirals.append(spiral)
    for i in range(n_examples):
        # Shuffle data into train and test sets
        subset = 'train' if i < n_train else 'test'
        fn = os.path.join(data_dir, subset,
                          'spiral_{:03d}.csv'.format(indices[i]))
        pd.DataFrame(spirals[indices[i]],
                     columns=['x', 'y', 'noisy_x', 'noisy_y']).\
                     to_csv(fn, index=False)

def test_dataset(data_dir='./spirals', subset='train', stats=False):
    print("Loading data...")
    dataset = SpiralsDataset(['spiral-x', 'spiral-y'], data_dir, subset)
    print("Testing batch collation...")
    data = seq_collate([dataset[i] for i in range(min(10, len(dataset)))])
    print("Batch shapes:")
    for d in data[:-2]:
        print(d.shape)
    print("Sequence lengths: ", data[-1])
    print("Checking through data for mismatched sequence lengths...")
    for i, data in enumerate(dataset):
        print("Sequence: ", dataset.seq_ids[i])
        x, y = data
        print(x.shape, y.shape)
        if len(x) != len(y):
            print("WARNING: Mismatched sequence lengths.")
    if stats:
        print("Statistics:")
        m_mean, m_std = dataset.mean_and_std()
        m_max, m_min = dataset.max_and_min()
        for m in modalities:
            print("--", m, "--")
            print("Mean:", m_mean[m])
            print("Std:", m_std[m])
            print("Max:", m_max[m])
            print("Min:", m_min[m])
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_examples', type=int, default=1000, metavar='N',
                        help='number of examples to generate (default: 1000)')
    parser.add_argument('--n_train', type=int, default=600, metavar='N',
                        help='number for training set (default: 600)')
    parser.add_argument('--timesteps', type=int, default=100, metavar='T',
                        help='number of timesteps per spiral (default: 100)')
    parser.add_argument('--data_dir', type=str, default='./spirals',
                        help='output directory (default: ./spirals)')
    parser.add_argument('--test', action='store_true', default=False,
                        help='test loading of dataset instead of generating')
    parser.add_argument('--subset', type=str, default="train",
                        help='whether to load train/test data')
    parser.add_argument('--stats', action='store_true', default=False,
                        help='whether to compute and print statistics')
    args = parser.parse_args()
    if args.test:
        test_dataset(args.data_dir, args.subset, args.stats)
    else:
        gen_dataset(args.n_examples, args.n_train,
                    args.timesteps, args.data_dir)
        
