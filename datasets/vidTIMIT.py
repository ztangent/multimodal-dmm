from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os, sys

import numpy as np
from PIL import Image

if __name__ == '__main__':
    from multiseq import MultiseqDataset, seq_collate
else:
    from .multiseq import MultiseqDataset, seq_collate

subjects = [
    'fadg0', 'faks0', 'fcft0', 'fcmh0', 'fcmr0', 'fcrh0', 'fdac1', 'fdms0',
    'fdrd1', 'fedw0', 'felc0', 'fgjd0', 'fjas0', 'fjem0', 'fjre0', 'fjwb0',
    'fkms0', 'fpkt0', 'fram1', 'mabw0', 'mbdg0', 'mbjk0', 'mccs0', 'mcem0',
    'mdab0', 'mdbb0', 'mdld0', 'mgwt0', 'mjar0', 'mjsw0', 'mmdb1', 'mmdm2',
    'mpdf0', 'mpgl0', 'mrcz0', 'mreb0', 'mrgg0', 'mrjo0', 'msjs1', 'mstk0',
    'mtas1', 'mtmr0', 'mwbt0'
]

class VidTIMITDataset(MultiseqDataset):
    """VidTIMIT audio/video/text dataset."""
    def __init__(self, data_dir, base_rate=None, item_as_dict=False):
        # Generate dataset if it doesn't exist yet
        if (not os.path.exists(data_dir) or
            len([f for f in os.listdir(data_dir) if f[-3:] == 'npy']) == 0):
            download_vidTIMIT(dest=data_dir)
        super(vidTIMITDataset, self).__init__(
            modalities=['video'], dirs=data_dir, regex="(\w+)_(\w+)\.npy",
            preprocess=None, rates=25, base_rate=base_rate, truncate=False,
            ids_as_mods=['person', 'action'], item_as_dict=item_as_dict)

def download_vidTIMIT(dest='./vidTIMIT'):
    """Downloads and preprocesses VidTIMIT dataset."""
    src_url = ('https://zenodo.org/record/158963/files/')
    
    import requests, zipfile
    from tqdm import tqdm

    def download(filename, source, dest):
        print("Downloading '{}'...".format(filename))
        url = source + filename
        try:
            with open(os.path.join(dest, filename), 'ab') as f:
                headers = {}
                pos = f.tell()
                if pos:
                    headers['Range'] = 'bytes={}-'.format(pos)
                resp = requests.get(url, headers=headers, stream=True)
                total_size = resp.headers.get('content-length', None)
                total = int(total_size)//1024 if total_size else None
                for data in tqdm(iterable=resp.iter_content(chunk_size=512),
                                 total=total, unit='KB'):
                    f.write(data)
        except requests.exceptions.RequestException:
            print("\nError downloading, attempting to resume...")
            download(filename, source, dest)

    # Use FFMPEG to crop from 180x144 to 128x128, then resize to 64x64
    ffmpeg_params = {'-s': '64x64',
                     '-vf': 'crop=128:128:26:8'}

    if not os.path.exists(dest):
        os.mkdir(dest)
    for subj in subjects:
        subj_path = os.path.join(dest, subj)
        zip_path = subj_path + '.zip'
        # Download and extract videos
        if not os.path.exists(zip_path):
            download(subj + '.zip', source=src_url, dest=dest)
        if not os.path.exists(subj_path):
            with zipfile.ZipFile(zip_path, "r") as f:
                print("Extracting subject '{}'".format(subj))
                f.extractall(dest)
        # Convert videos to NPY
        vid_subdir = os.path.join(subj_path, 'video')
        for vid_name in os.listdir(vid_subdir):
            vid_path = os.path.join(vid_subdir, vid_name)
            print("Converting {} to NPY...".format(vid_path))
            vid_data = img_dir_to_npy(vid_path)
            vid_data = preprocess_video(vid_data)
            npy_path = os.path.join(vid_subdir, vid_name + '.npy')
            np.save(npy_path, vid_data)

def img_dir_to_npy(path):
    fnames = sorted(os.listdir(path))
    npy = np.array([np.array(Image.open(os.path.join(path, fname)))
                    for fname in fnames])
    return npy
            
def preprocess_video(video):
    """Crop, normalize to [0,1] and swap dimensions."""
    import skimage.transform
    height, width = video.shape[1:3]
    side = min(height, width)
    x0 = (width - side)//2
    y0 = (height - side)//2
    # Crop to central square
    video = np.array(video[:, y0:y0+side, x0:x0+side])
    # Resize to 64 by 64
    video = np.stack([skimage.transform.resize(video[t], (64, 64, 3))
                     for t in range(video.shape[0])], axis=0)
    # Transpose to (time, channels, rows, cols)
    video = np.transpose(video, (0,3,1,2))
    print(video.shape, video.max(), video.min())
    return video            
            
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./vidTIMIT',
                        help='data directory (default: ./vidTIMIT)')
    parser.add_argument('--stats', action='store_true', default=False,
                        help='whether to compute and print statistics')
    args = parser.parse_args()
    download_vidTIMIT(args.data_dir)
