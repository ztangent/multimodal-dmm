from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os, sys

import numpy as np
import skvideo.io

if __name__ == '__main__':
    from multiseq import MultiseqDataset, seq_collate
else:
    from .multiseq import MultiseqDataset, seq_collate

actions = {
    'bend'  : 'Bend',
    'jack'  : 'Jumping jack',
    'jump'  : 'Jump',
    'pjump' : 'Jump in place',
    'run'   : 'Run',
    'side'  : 'Gallop sideways',
    'skip'  : 'Skip',
    'walk'  : 'Walk',
    'wave1' : 'One-hand wave',
    'wave2' : 'Two-hands wave'
}

duplicates = ['lena_walk', 'lena_run', 'lena_skip']

class WeizmannDataset(MultiseqDataset):
    """Weizmann human action video dataset."""
    def __init__(self, data_dir, base_rate=None, item_as_dict=False):
        # Generate dataset if it doesn't exist yet
        if (not os.path.exists(data_dir) or
            len([f for f in os.listdir(data_dir) if f[-3:] == 'npy']) == 0):
            download_weizmann(dest=data_dir)
        super(WeizmannDataset, self).__init__(
            modalities=['video'], dirs=data_dir, regex="(\w+)_(\w+)\.npy",
            preprocess=None, rates=25, base_rate=base_rate, truncate=False,
            ids_as_mods=['person', 'action'], item_as_dict=item_as_dict)

def download_weizmann(dest='./weizmann'):
    """Downloads and preprocesses Weizmann human action dataset."""
    src_url = ('http://www.wisdom.weizmann.ac.il/~vision/' +
               'VideoAnalysis/Demos/SpaceTimeActions/DB/')
    
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def progress(count, blockSize, totalSize):
        percent = int(count*blockSize*100/totalSize)
        sys.stdout.write("\rProgress: {:2d}%".format(percent))
        sys.stdout.flush()

    def download(filename, source, dest):
        print("Downloading '{}'...".format(filename))
        urlretrieve(source + filename, dest + filename, reporthook=progress)
    
    import zipfile
    for act in actions.keys():
        zip_path = os.path.join(dest, act + '.zip')
        if not os.path.exists(zip_path):
            download(act + '.zip', source=src_url, dest=dest)
        with zipfile.ZipFile(zip_path, "r") as f:
            vid_names = [vn for vn in f.namelist() if vn[-3:] == 'avi']
            print("Extracting '{}' videos... ({} files)".\
                  format(act, len(vid_names)))
            f.extractall(dest, members=vid_names)
        for vn in vid_names:
            # Skip duplicate videos (e.g. 'lena_walk2.avi')
            if vn[:-5] in duplicates and vn[-5] == '2':
                continue
            print("Converting {} to NPY...".format(vn))
            vid_path = os.path.join(dest, vn)
            vid_data = skvideo.io.vread(vid_path)
            # Rename original of duplicate pairs ('lena_walk1'->'lena_walk')
            if vn[:-5] in duplicates:
                vid_path = vid_path[:-5] + vid_path[-4:]
            np.save(vid_path[:-3] + 'npy', vid_data)

def test_dataset(data_dir='./weizmann', stats=False):
    print("Loading data...")
    dataset = WeizmannDataset(data_dir)
    print("Sequence ID values:")
    for s in dataset.seq_id_sets:
        print(s)
    print("Testing batch collation...")
    data = seq_collate([dataset[i] for i in range(min(10, len(dataset)))])
    print("Batch shapes:")
    for d in data[:-2]:
        print(d.shape)
    print("Sequence lengths: ", data[-1])
    print("Checking through data for mismatched sequence lengths...")
    for i, data in enumerate(dataset):
        print("Sequence: ", dataset.seq_ids[i])
        video, person, action = data
        print(video.shape, person.shape, action.shape)
        if len(video) != len(person) or len(video) != len(action):
            print("WARNING: Mismatched sequence lengths.")
    if stats:
        print("Statistics:")
        m_mean, m_std = dataset.mean_and_std()
        m_max, m_min = dataset.max_and_min()
        for m in ['video', 'person', 'action']:
            print("--", m, "--")
            print("Mean:", m_mean[m])
            print("Std:", m_std[m])
            print("Max:", m_max[m])
            print("Min:", m_min[m])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./weizmann',
                        help='data directory (default: ./weizmann)')
    parser.add_argument('--stats', action='store_true', default=False,
                        help='whether to compute and print statistics')
    args = parser.parse_args()
    test_dataset(args.data_dir, args.stats)
