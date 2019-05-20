from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os, re, copy, itertools

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class MultiseqDataset(Dataset):
    """Multimodal dataset for (synchronous) time series and sequential data."""
    
    def __init__(self, modalities, dirs, regex, preprocess, rates,
                 base_rate=None, truncate=False,
                 ids_as_mods=[], item_as_dict=False):
        """Loads valence ratings and features for each modality.

        modalities -- names of each input modality
        dirs -- list of directories containing input features
        regex -- regex patterns for the filenames of each modality
        preprocess -- data pre-processing functions for pandas dataframes
        rates -- sampling rates of each modality
        base_rate -- base_rate to subsample/ovesample to
        truncate -- if true, truncate to modality with minimum length
        ids_as_mods -- add sequence ids as modalities with these names
        item_as_dict -- whether to return data as dictionary
        """
        # Store arguments
        self.modalities = modalities
        if type(rates) is not list:
            self.rates = [rates] * len(modalities)
        else:
            self.rates = rates
        self.base_rate = base_rate if base_rate != None else min(self.rates)
        self.item_as_dict = item_as_dict

        # Convert to modality-indexed dictionaries
        if type(dirs) is not list:
            dirs = [dirs] * len(self.modalities)
        dirs = {m: d for m, d in zip(modalities, dirs)}
        if type(regex) is not list:
            regex = [regex] * len(self.modalities)
        regex = {m: r for m, r in zip(modalities, regex)}
        if preprocess is None:
            preprocess = lambda x : x
        if type(preprocess) is not list:
            preprocess = [preprocess] * len(self.modalities)
        preprocess = {m: p for m, p in zip(modalities, preprocess)}
        
        # Load filenames into lists and extract regex-captured sequence IDs
        paths = dict()
        seq_ids = dict()
        for m in modalities:
            paths[m] = []
            seq_ids[m] = []
            for fn in os.listdir(dirs[m]):
                match = re.match(regex[m], fn)
                if not match:
                    continue
                paths[m].append(os.path.join(dirs[m], fn))
                seq_ids[m].append(match.groups())
            # Sort by values of captured indices
            paths[m] = [p for _, p in sorted(zip(seq_ids[m], paths[m]))]
            seq_ids[m].sort()

        # Check that number and IDs of files/sequences are matched
        self.seq_ids = seq_ids[modalities[0]]
        for m in modalities:
            if len(paths[m]) != len(self.seq_ids):
                raise Exception("Number of files ({}) do not match.".\
                                format(len(paths[m])))
            if seq_ids[m] != self.seq_ids:
                raise Exception("Sequence IDs do not match.")
        # Store all possible values for sequence IDs
        self.seq_id_sets = [list(sorted(set(s_ids))) for s_ids in
                            list(zip(*self.seq_ids))]

        # Compute ratio to base rate
        self.ratios = {m: r/self.base_rate for m, r in
                       zip(self.modalities, self.rates)}
            
        # Load data from files
        self.data = {m: [] for m in modalities}
        self.orig = {m: [] for m in modalities}
        self.lengths = []
        for i in range(len(self.seq_ids)):
            seq_len = float('inf')
            # Load each input modality
            for m, data in self.data.iteritems():
                fp = paths[m][i]
                if re.match("^.*\.npy", fp):
                    # Load as numpy array
                    d = np.load(fp)
                elif re.match("^.*\.(csv|txt)", fp):
                    # Use pandas to read and pre-process CSV files
                    d = pd.read_csv(fp)
                    d = np.array(preprocess[m](d))
                elif re.match("^.*\.tsv", fp):
                    d = pd.read_csv(fp, sep='\t')
                    d = np.array(preprocess[m](d))
                # Store original data before resampling
                self.orig[m].append(d)
                # Subsample/oversample data to base rate
                ratio = self.ratios[m]
                if ratio > 1:
                    # Time average so that data is at base rate
                    ratio = int(ratio)
                    end = ratio * (len(d)//ratio)
                    avg = np.mean(d[:end].reshape(-1, ratio, *d.shape[1:]), 1)
                    if end < len(d):
                        remain = d[end:].mean(axis=0)[np.newaxis,:]
                        d = np.concatenate([avg, remain])
                    else:
                        d = avg
                else:
                    # Repeat so that data is at base rate
                    ratio = int(1. / ratio)
                    d = np.repeat(d, ratio, axis=0)
                data.append(d)
                if len(d) < seq_len:
                    seq_len = len(d)
            # Truncate to minimum sequence length
            if truncate:
                for m in self.modalities:
                    self.data[m][-1] = self.data[m][-1][:seq_len]
            self.lengths.append(seq_len)

        # Add information from sequence IDs as additional modalities
        self.ids_as_mods = ids_as_mods
        for m in ids_as_mods:
            self.modalities.append(m)
            self.rates.append(self.base_rate)
            self.ratios[m] = 1.0
            self.data[m] = []
            self.orig[m] = []
        for seq_id, seq_len in zip(self.seq_ids, self.lengths):
            for k, m in enumerate(ids_as_mods):
                # Ignore ID fields that are set to None
                if m is None:
                    continue
                # Repeat ID field for the length of the whole sequence
                d = self.seq_id_sets[k].index(seq_id[k])
                d = np.array([[d]] * seq_len)
                self.data[m].append(d)
                self.orig[m].append(d)
            
    def __len__(self):
        return len(self.seq_ids)

    def __getitem__(self, i):
        if self.item_as_dict:
            d = {m: self.data[m][i] for m in self.modalities}
            d['length'] = self.lengths[i]
            return d
        else:
            return tuple(self.data[m][i] for m in self.modalities)

    def mean_and_std(self, modalities=None):
        """Compute mean+std across time and samples for given modalities."""
        if modalities is None:
            modalities = self.modalities
        m_mean = {m: np.nanmean(np.concatenate(self.data[m], 0), axis=0)
                  for m in modalities}
        m_std = {m: np.nanstd(np.concatenate(self.data[m], 0), axis=0)
                 for m in modalities}
        return m_mean, m_std

    def max_and_min(self, modalities=None):
        """Compute max+min across time and samples for given modalities."""
        if modalities is None:
            modalities = self.modalities
        m_max = {m: np.nanmax(np.stack([a.max(0) for a in self.data[m]]), 0)
                 for m in modalities}
        m_min = {m: np.nanmin(np.stack([a.min(0) for a in self.data[m]]), 0)
                 for m in modalities}
        return m_max, m_min
    
    def normalize_(self, modalities=None, method='meanvar', ref_data=None):
        """Normalize data either by mean-and-variance or to [-1,1])."""
        if modalities is None:
            # Default to all modalities
            modalities = self.modalities
        if ref_data is None:
            # Default to computing stats over self
            ref_data = self
        if method == 'range':
            # Range normalization
            m_max, m_min = ref_data.max_and_min(modalities)
            # Compute range per dim and add constant to ensure it is non-zero
            m_rng = {m: (m_max[m]-m_min[m]) for m in modalities}
            m_rng = {m: m_rng[m] * (m_rng[m] > 0) + 1e-10 * (m_rng[m] <= 0)
                     for m in modalities}
            for m in modalities:
                self.data[m] = [(a-m_min[m]) / m_rng[m] * 2 - 1 for
                                a in self.data[m]]
        else:
            # Mean-variance normalization
            m_mean, m_std = ref_data.mean_and_std(modalities)
            for m in modalities:
                self.data[m] = [(a-m_mean[m]) / (m_std[m] + 1e-10) for
                                a in self.data[m]]

    def normalize(self, modalities=None, method='meanvar', ref_data=None):
        """Normalize data (returns new dataset)."""
        dataset = copy.deepcopy(self)
        dataset.normalize_(modalities, method, ref_data)
        return dataset
            
    def split_(self, n, n_is_len=False):
        """Splits each sequence into chunks (in place)."""
        if n_is_len:
            # Use n as maximum chunk length
            split = [range(n, l, n) for l in self.lengths]
        else:
            # Use n as number of chunks
            split = [n for l in self.lengths]
        for m in self.modalities:
            self.data[m] = list(itertools.chain.from_iterable(
                [np.array_split(a, s, 0) for a,s in zip(self.data[m], split)]))
        if n_is_len:
            self.seq_ids = list(itertools.chain.from_iterable(
                [[i] * (len(s)+1) for i,s in zip(self.seq_ids, split)]))
        else:
            self.seq_ids = list(itertools.chain.from_iterable(
                [[i] * n for i in self.seq_ids]))            
        self.lengths = [len(d) for d in self.data[self.modalities[0]]]

    def split(self, n, n_is_len=False):
        """Splits each sequence into chunks (returns new dataset)."""
        dataset = copy.deepcopy(self)
        dataset.split_(n, n_is_len)
        return dataset

    def select(self, seq_ids, invert=False):
        """Select sequences by identifiers and return new dataset.

        seq_ids -- list of lists, where list k contains the
                   identifiers extracted by the kth regex group
        invert -- whether to invert selection (i.e. delete specified IDs)
        """
        sel = copy.deepcopy(self)
        # Find indices in the intersection of all specified identifiers
        idx = list(range(len(self)))
        for k in range(len(seq_ids)):
            if seq_ids[k] is None:
                seq_ids[k] = self.seq_id_sets[k]
            idx = [i for i, seq_id in enumerate(self.seq_ids)
                   if seq_id[k] in seq_ids[k] and i in idx]
        if invert:
            idx = [i for i in range(len(self)) if i not in idx]
        # Select data
        sel.seq_ids = [sel.seq_ids[i] for i in idx]
        sel.seq_id_sets = [list(sorted(set(s_ids))) for s_ids in
                           list(zip(*sel.seq_ids))]
        sel.lengths = [sel.lengths[i] for i in idx]
        for m in self.modalities:
            sel.data[m] = [sel.data[m][i] for i in idx]
            sel.orig[m] = [sel.orig[m][i] for i in idx]
        return sel
    
    @classmethod
    def merge(cls, set1, set2):
        """Merge two datasets."""
        if (set1.modalities != set2.modalities):
            raise Exception("Modalities need to match.")
        if (set1.base_rate != set2.base_rate):
            raise Exception("Base rates need to match.")
        merged = copy.deepcopy(set1)
        merged.orig.clear()
        merged.seq_ids += set2.seq_ids
        merged.seq_id_sets = [list(set(set1.seq_id_sets[k]) |
                                   set(set2.seq_id_sets[k]))
                              for k in range(len(set1.seq_id_sets))]
        merged.rates = [merged.base_rate] * len(merged.modalities)
        merged.ratios = [1] * len(merged.modalities)
        for m in merged.modalities:
            merged.data[m] += copy.deepcopy(set2.data[m])
        return merged
        
def len_to_mask(lengths, time_first=True):
    """Converts list of sequence lengths to a mask tensor."""
    mask = torch.arange(max(lengths)).expand(len(lengths), max(lengths))
    mask = mask < torch.tensor(lengths).unsqueeze(1)
    if time_first:
        mask = mask.transpose(0, 1)
    return mask.unsqueeze(-1)

def mask_to_extent(mask, time_first=True):
    """Return first and last observed indices given mask tensor."""
    if not time_first:
        mask = mask.transpose(0, 1)
    t_max, b_dim = mask.shape[0:1]
    idx = torch.arange(t_max).expand(b_dim, t_max).transpose(0, 1)
    idx = mask.view(t_max, b_dim).long() * idx
    t_start = idx.min(dim=0).tolist()
    t_stop = idx.max(dim=0).tolist()
    return t_start, t_stop

def pad_and_merge(sequences, max_len=None):
    """Pads and merges unequal length sequences into batch tensor."""
    dims = sequences[0].shape[1:]
    lengths = [len(seq) for seq in sequences]
    if max_len is None:
        max_len = max(lengths)
    padded_seqs = torch.ones(max_len, len(sequences), *dims) * float('nan')
    for i, seq in enumerate(sequences):
        end = lengths[i]
        padded_seqs[:end, i] = torch.from_numpy(seq[:end])
    if len(sequences) == 1:
        padded_seqs = padded_seqs.float()
    return padded_seqs

def seq_collate(data, time_first=True):
    """Collates multimodal variable length sequences into padded batch."""
    padded = []
    n_modalities = len(data)
    lengths = np.zeros(n_modalities, dtype=int)
    data.sort(key=lambda x: len(x[0]), reverse=True)
    data = zip(*data)
    for modality in data:
        m_lengths = [len(seq) for seq in modality]
        lengths = np.maximum(lengths, m_lengths)
    lengths = list(lengths)
    for modality in data:
        m_padded = pad_and_merge(modality, max(lengths))
        padded.append(m_padded if time_first else m_padded.transpose(0, 1))
    mask = len_to_mask(lengths)
    if not time_first:
        mask = mask.transpose(0, 1)
    return tuple(padded + [mask, lengths])

def seq_collate_dict(data, time_first=True):
    """Collate that accepts and returns dictionaries of batch tensors."""
    batch = {}
    modalities = [k for k in data[0].keys() if  k != 'length']
    order = sorted(range(len(data)),
                   key=lambda i: data[i]['length'], reverse=True)
    data.sort(key=lambda d: d['length'], reverse=True)
    lengths = [d['length'] for d in data]
    for m in modalities:
        m_data = [d[m] for d in data]
        m_padded = pad_and_merge(m_data, max(lengths))
        batch[m] = m_padded if time_first else m_padded.transpose(0, 1)
    mask = len_to_mask(lengths)
    if time_first:
        mask = mask.transpose(0, 1)
    return batch, mask, lengths, order

def seq_decoll(batch, lengths, order, time_first=True):
    """Decollate batched data by de-padding and reordering."""
    if time_first:
        data = [batch[:lengths[idx],idx].cpu().numpy() for idx in order]
    else:
        data = [batch[idx,:lengths[idx]].cpu().numpy() for idx in order]
    return data

def seq_decoll_dict(batch_dict, lengths, order, time_first=True):
    """Decollate dictionary of batch tensors into dictionary of lists"""
    return {k: seq_decollate(batch, lengths, order, time_first)
            for k, batch in batch_dict.iteritems()}

def func_delete(batch_in, del_func, lengths=None, modalities=None):
    """Use del_func to compute time indices to delete. Assumes time_first."""
    if modalities == None:
        modalities = batch_in.keys()
    batch_out = dict()
    for m in batch_in.keys():
        batch_out[m] = torch.tensor(batch_in[m])
        if m not in modalities:
            continue
        t_max, b_dim = batch_in[m].shape[:2]
        if lengths == None:
            lengths = [t_max] * b_dim
        for b in range(b_dim):
            del_idx = del_func(lengths[b])
            batch_out[m][del_idx, b] = float('nan')
    return batch_out

def rand_delete(batch_in, del_frac, lengths=None, modalities=None):
    """Introduce random memoryless errors / deletions into a data batch"""
    def del_func(length):
        return np.random.choice(length, int(del_frac * length), False)
    return func_delete(batch_in, del_func, lengths, modalities)

def burst_delete(batch_in, burst_frac, lengths=None, modalities=None):
    """Introduce random burst errors / deletions into a data batch"""
    def del_func(length):
        t_start = np.random.randint(length)
        t_stop = min(t_start + int(burst_frac * length), length)
        return range(t_start, t_stop)
    return func_delete(batch_in, del_func, lengths, modalities)

def keep_segment(batch_in, f_start, f_stop, lengths=None, modalities=None):
    """Delete all data outside of specified time fraction [f_start, f_stop)."""
    def del_func(length):
        t_start, t_stop = int(f_start * length), int(f_stop * length)
        return range(0, t_start) + range(t_stop, length)
    return func_delete(batch_in, del_func, lengths, modalities)

def del_segment(batch_in, f_start, f_stop, lengths=None, modalities=None):
    """Delete specified time fraction [f_start, f_stop)."""
    def del_func(length):
        t_start, t_stop = int(f_start * length), int(f_stop * length)
        return range(t_start, t_stop)
    return func_delete(batch_in, del_func, lengths, modalities)
    
        
