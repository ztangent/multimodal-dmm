from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import requests
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
