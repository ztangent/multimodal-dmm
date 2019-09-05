from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os, sys

import numpy as np

if __name__ == '__main__':
    from multiseq import MultiseqDataset, seq_collate
else:
    from .multiseq import MultiseqDataset, seq_collate

fps = 25.0    
subjects = [
    'fadg0', 'faks0', 'fcft0', 'fcmh0', 'fcmr0', 'fcrh0', 'fdac1', 'fdms0',
    'fdrd1', 'fedw0', 'felc0', 'fgjd0', 'fjas0', 'fjem0', 'fjre0', 'fjwb0',
    'fkms0', 'fpkt0', 'fram1', 'mabw0', 'mbdg0', 'mbjk0', 'mccs0', 'mcem0',
    'mdab0', 'mdbb0', 'mdld0', 'mgwt0', 'mjar0', 'mjsw0', 'mmdb1', 'mmdm2',
    'mpdf0', 'mpgl0', 'mrcz0', 'mreb0', 'mrgg0', 'mrjo0', 'msjs1', 'mstk0',
    'mtas1', 'mtmr0', 'mwbt0'
]

class VidTIMITDataset(MultiseqDataset):
    """VidTIMIT audio/video dataset."""
    def __init__(self, data_dir, base_rate=None, item_as_dict=False):
        # Generate dataset if it doesn't exist yet
        audio_dir = os.path.join(data_dir, 'audio')
        video_dir = os.path.join(data_dir, 'video')
        if (not os.path.exists(data_dir) or
            not os.path.exists(audio_dir) or
            not os.path.exists(video_dir) or
            len([f for f in os.listdir(audio_dir) if f[-3:] == 'npy']) == 0 or
            len([f for f in os.listdir(video_dir) if f[-3:] == 'npy']) == 0):
            download_vidTIMIT(dest=data_dir)
            
        super(VidTIMITDataset, self).__init__(
            modalities=['audio', 'video'], dirs=[audio_dir, video_dir],
            regex="(\w+)_(\w+)\.npy", preprocess=None,
            rates=fps, base_rate=base_rate, truncate=True,
            ids_as_mods=[], item_as_dict=item_as_dict)

def download_vidTIMIT(dest='./vidTIMIT'):
    """Downloads and preprocesses VidTIMIT dataset."""
    src_url = ('https://zenodo.org/record/158963/files/')
    
    import requests, zipfile
    from tqdm import tqdm
    import PIL.Image, skimage.transform
    import scipy.io.wavfile, scipy.signal

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

    def img_dir_to_npy(path):
        fnames = sorted(os.listdir(path))
        npy = np.array([np.array(PIL.Image.open(os.path.join(path, fname)))
                        for fname in fnames])
        return npy

    def preprocess_video(video):
        """Crop, normalize to [0,1] and swap dimensions."""
        height, width = video.shape[1:3]
        side = min(height, width)
        x0 = (width - side)//2
        y0 = (height - side)//2
        # Crop to central square
        video = np.array(video[:, y0:y0+side, x0:x0+side])
        # Resize to 64 by 64 and normalize to [0, 1]
        video = np.stack([skimage.transform.resize(video[t], (64, 64, 3))
                         for t in range(video.shape[0])], axis=0)
        # Transpose to (time, channels, rows, cols)
        video = np.transpose(video, (0,3,1,2))
        return video            

    def preprocess_audio(audio, rate):
        """Convert to spectrogram using 25 windows per second."""
        win_sz = rate / fps * 2
        # Perform Short Time Fourier Transform (STFT)
        f, t, spec = scipy.signal.stft(audio, rate,
                                       nperseg=win_sz, noverlap=win_sz/2)
        # Swap time and frequency axes
        spec = spec.T
        # Stack the windows [T-2, T-1, T, T+1, T+2] as channels for Tth feature
        overlap = 2
        n_wins = spec.shape[0]
        spec = np.pad(spec, [(overlap, overlap), (0, 0)], mode='constant')
        spec = spec[np.arange(n_wins)[:, None] + np.arange(overlap*2+1)]
        # Separate and concat real and imaginary parts as channels
        spec = np.concatenate([np.real(spec), np.imag(spec)], axis=1)
        return spec

    # Create dataset directories
    if not os.path.exists(dest):
        os.mkdir(dest)
    vid_dir = os.path.join(dest, 'video')
    if not os.path.exists(vid_dir):
        os.mkdir(vid_dir)
    aud_dir = os.path.join(dest, 'audio')
    if not os.path.exists(aud_dir):
        os.mkdir(aud_dir)
        
    for subj in subjects:
        subj_path = os.path.join(dest, subj)

        # Download and extract videos
        zip_path = subj_path + '.zip'
        if not os.path.exists(zip_path):
            download(subj + '.zip', source=src_url, dest=dest)
        if not os.path.exists(subj_path):
            with zipfile.ZipFile(zip_path, "r") as f:
                print("Extracting subject '{}'".format(subj))
                f.extractall(dest)

        # Convert videos to NPY
        subj_vid_dir = os.path.join(subj_path, 'video')
        for vid_name in os.listdir(subj_vid_dir):
            # Skip non-video items
            vid_path = os.path.join(subj_vid_dir, vid_name)
            if not os.path.isdir(vid_path):
                continue
            # Skip head rotation videos
            if vid_name[:4] == 'head':
                continue
            print("Converting {} to NPY...".format(vid_path))
            vid_data = img_dir_to_npy(vid_path)
            vid_data = preprocess_video(vid_data)
            # Save in main video directory
            npy_path = os.path.join(vid_dir, subj + '_' + vid_name + '.npy')
            np.save(npy_path, vid_data)

        # Convert audio waveforms to spectrogram NPY files
        subj_aud_dir = os.path.join(subj_path, 'audio')
        for aud_name in os.listdir(subj_aud_dir):
            # Skip non-WAV files
            if aud_name[-4:] != '.wav':
                continue
            aud_path = os.path.join(subj_aud_dir, aud_name)
            print("Converting {} to NPY...".format(aud_path))
            rate, aud_data = scipy.io.wavfile.read(aud_path)
            aud_data = preprocess_audio(aud_data, rate)
            aud_name = aud_name[:-4]
            # Save in main audio directory
            npy_path = os.path.join(aud_dir, subj + '_' + aud_name + '.npy')
            np.save(npy_path, aud_data)

def test_dataset(data_dir='./vidTIMIT', stats=False):
    print("Loading data...")
    dataset = VidTIMITDataset(data_dir)
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
        audio, video = data
        print(audio.shape, video.shape)
        if len(audio) != len(video):
            print("WARNING: Mismatched sequence lengths.")
    if stats:
        print("Statistics:")
        m_mean, m_std = dataset.mean_and_std()
        m_max, m_min = dataset.max_and_min()
        for m in ['audio', 'video']:
            print("--", m, "--")
            print("Mean:", m_mean[m])
            print("Std:", m_std[m])
            print("Max:", m_max[m])
            print("Min:", m_min[m])
            
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./vidTIMIT',
                        help='data directory (default: ./vidTIMIT)')
    parser.add_argument('--stats', action='store_true', default=False,
                        help='whether to compute and print statistics')
    args = parser.parse_args()
    test_dataset(args.data_dir, args.stats)
