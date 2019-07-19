from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os, sys

import numpy as np

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
        urlretrieve(source + filename, os.path.join(dest, filename),
                    reporthook=progress)
        print('')

    # Use FFMPEG to crop from 180x144 to 128x128, then resize to 64x64
    ffmpeg_params = {'-s': '64x64',
                     '-vf': 'crop=128:128:26:8'}

    import zipfile
    if not os.path.exists(dest):
        os.mkdir(dest)
    for subj in subjects:
        zip_path = os.path.join(dest, subj + '.zip')
        if not os.path.exists(zip_path):
            download(subj + '.zip', source=src_url, dest=dest)
        with zipfile.ZipFile(zip_path, "r") as f:
            print("Extracting subject '{}'".format(subj))
            f.extractall(dest)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./vidTIMIT',
                        help='data directory (default: ./vidTIMIT)')
    parser.add_argument('--stats', action='store_true', default=False,
                        help='whether to compute and print statistics')
    args = parser.parse_args()
    download_vidTIMIT(args.data_dir)
