from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from builtins import range
import os, sys

import numpy as np

if __name__ == '__main__':
    from multiseq import MultiseqDataset, seq_collate
else:
    from .multiseq import MultiseqDataset, seq_collate

fps = 25.0
persons = ['daria', 'denis', 'eli', 'ido', 'ira',
           'lena', 'lyova', 'moshe', 'shahar']
actions = ['bend', 'jack', 'jump', 'pjump', 'run',
           'side', 'skip', 'walk', 'wave1', 'wave2']
descriptions = ['Bend', 'Jumping jack', 'Jump',
                'Jump in place', 'Run' 'Gallop sideways',
                'Skip', 'Walk', 'One-hand wave', 'Two-hand wave']

duplicates = ['lena_walk', 'lena_run', 'lena_skip']

class WeizmannDataset(MultiseqDataset):
    """Weizmann human action video dataset."""
    def __init__(self, data_dir, base_rate=None, item_as_dict=False):
        # Generate dataset if it doesn't exist yet
        if (not os.path.exists(data_dir) or
            len([f for f in os.listdir(data_dir) if f[-3:] == 'npy']) == 0):
            download_weizmann(dest=data_dir)
        super(WeizmannDataset, self).__init__(
            modalities=['video', 'mask'], dirs=data_dir,
            regex=["([^_\W]+)_([^_\W]+)\.npy",
                   "([^_\W]+)_([^_\W]+)_mask\.npy"],
            preprocess=None, rates=25, base_rate=base_rate, truncate=False,
            ids_as_mods=['person', 'action'], item_as_dict=item_as_dict)

def download_weizmann(dest='./weizmann'):
    """Downloads and preprocesses Weizmann human action dataset."""
    src_url = ('http://www.wisdom.weizmann.ac.il/~vision/' +
               'VideoAnalysis/Demos/SpaceTimeActions/DB/')

    if __name__ == '__main__':
        import utils
    else:
        from . import utils
    import zipfile, scipy.io, skvideo.io

    # Use FFMPEG to crop from 180x144 to 128x128, then resize to 64x64
    ffmpeg_params = {'-s': '64x64',
                     '-vf': 'crop=128:128:26:8'}

    # Download masks / silhouettes
    if not os.path.exists(dest):
        os.mkdir(dest)
    if not os.path.exists(os.path.join(dest, 'classification_masks.mat')):
        utils.download('classification_masks.mat', source=src_url, dest=dest)
    masks = scipy.io.loadmat(os.path.join(dest, 'classification_masks.mat'))
    masks = masks['original_masks'][0,0]

    # Download videos for each action
    for act in actions:
        zip_path = os.path.join(dest, act + '.zip')
        if not os.path.exists(zip_path):
            utils.download(act + '.zip', source=src_url, dest=dest)
        with zipfile.ZipFile(zip_path, "r") as f:
            vid_names = [vn for vn in f.namelist() if vn[-3:] == 'avi']
            print("Extracting '{}' videos... ({} files)".\
                  format(act, len(vid_names)))
            f.extractall(dest, members=vid_names)
        for vn in vid_names:
            # Remove extension
            vn_no_ext = vn[:-4]
            # Skip duplicate videos (e.g. 'lena_walk2.avi')
            if vn_no_ext[:-1] in duplicates and vn_no_ext[-1] == '2':
                continue
            print("Converting {} to NPY...".format(vn))
            vid_path = os.path.join(dest, vn)
            vid_data = skvideo.io.vread(vid_path, outputdict=ffmpeg_params)
            vid_data = preprocess_video(vid_data)
            mask_data = preprocess_mask(masks[vn_no_ext])
            # Rename original of duplicate pairs ('lena_walk1'->'lena_walk')
            if vn_no_ext[:-1] in duplicates:
                vn_no_ext = vn_no_ext[:-1]
            npy_path = os.path.join(dest, vn_no_ext + '.npy')
            np.save(npy_path, vid_data)
            print("Saving masks for {} to NPY...".format(vn))
            npy_path = os.path.join(dest, vn_no_ext + '_mask.npy')
            np.save(npy_path, mask_data)

def preprocess_video(video):
    """Crop, normalize to [0,1] and swap dimensions."""
    height, width = video.shape[1:3]
    side = min(height, width)
    x0 = (width - side)//2
    y0 = (height - side)//2
    # Crop to central square
    video = np.array(video[:, y0:y0+side, x0:x0+side])
    # Transpose to (time, channels, rows, cols)
    video = np.transpose(video, (0,3,1,2))
    # Scale from [0, 255] to [0, 1]
    video = video / 255.0
    return video

def preprocess_mask(mask):
    """Crop, normalize and swap dimensions."""
    import skimage.transform
    height, width = mask.shape[0:2]
    side = min(height, width)
    x0 = (width - side)//2
    y0 = (height - side)//2
    # Crop to central square, convert to float
    mask = np.array(mask[y0:y0+side, x0:x0+side, :]).astype(np.float64)
    # Transpose to (time, rows, cols)
    mask = np.transpose(mask, (2,0,1))
    # Resize to 64 by 64
    mask = np.stack([skimage.transform.resize(mask[t], (64, 64))
                     for t in range(mask.shape[0])], axis=0)
    # Add channels dimension
    mask = mask[:, np.newaxis, :, :]
    return mask

def test_dataset(data_dir='./weizmann', stats=False):
    print("Loading data...")
    dataset = WeizmannDataset(data_dir)
    print("Number of sequences:", len(dataset))
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
        video, mask, person, action = data
        print(video.shape, mask.shape, person.shape, action.shape)
        if (len(video) != len(person) or
            len(video) != len(action) or
            len(video) != len(mask)):
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
