import os
import argparse

import numpy as np
import numpy.random as r
import pandas as pd

def gen_spiral(start_r, stop_r, start_theta, stop_theta,
               aspect_ratio=1, timesteps=100):
    r = np.linspace(start_r, stop_r, timesteps)
    theta = np.linspace(start_theta, stop_theta, timesteps)
    x = aspect_ratio * r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_examples', type=int, default=1000, metavar='N',
                        help='number of examples to generate (default: 1000)')
    parser.add_argument('--timesteps', type=int, default=100, metavar='T',
                        help='number of timesteps per spiral (default: 100)')
    parser.add_argument('--output_dir', type=str, default='./spirals',
                        help='output directory (default: ./spirals)')
    args = parser.parse_args()
    # Create output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    r.seed(1)
    # Shuffle indices
    indices = list(range(args.n_examples))
    r.shuffle(indices)
    # Generate spirals
    for i in range(args.n_examples):
        # First half are CW spirals, second half are CCW spirals
        direction = 1 if (i >= args.n_examples/2) else -1
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
                          aspect_ratio, args.timesteps)
        # Add Gaussian noise
        x += 0.1 * r.randn(args.timesteps)
        y += 0.1 * r.randn(args.timesteps)
        # Save data with random index
        spiral = np.stack([x, y], axis=1)
        fn = os.path.join(args.output_dir,
                          'spiral_{:03d}.csv'.format(indices[i]))
        pd.DataFrame(spiral, columns=['x', 'y']).to_csv(fn, index=False)
        
