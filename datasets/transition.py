
import os

import torch
from torch.utils.data import Dataset, Subset, Sampler, BatchSampler, SequentialSampler, RandomSampler, DistributedSampler

from collections import namedtuple
from typing import Optional, List, Iterator
import random
import csv


T_CSV = namedtuple("T_CSV", ["input", "output", "variation", "source", "target", "split"])

class TransitionDataset(Dataset):
    """
    Wrapper for dataset adding transitions between images with similar features.
    Supposes that dataset contains the following attributes:
        :attr split: (str) Train, val, or test. The current split to load.
        :attr root: (str) Root directory where data is stored.
        :attr base_folder: (str) Folder for the current dataset.
        :attr indices: (List[str]) List matching the partition ids with the dataset ids
    Attributes can be passed as arguments if not provided by the dataset. By default, takes  
    the members of the dataset but priority can be given to the wrapper arguments by setting 
    :override_args: to True. Member names may not correspond to the ones given, they cane be
    renamed using [attr]_alias arguments.
    """

    def __init__(self, 
                dataset: Dataset, 
                num_variations: int = 40, 
                split: str = "train", 
                root: str = "Data/", 
                base_folder: str = "celeba", 
                indices: List[str] = None, 
                split_alias: str = "split",
                root_alias: str = "root",
                base_folder_alias: str = "base_folder",
                indices_alias: str = "indices",
                override_args : bool = False, 
                **kwargs):
        
        super(TransitionDataset, self).__init__(**kwargs)

        self.dataset = dataset
        self.split = split if not hasattr(self.dataset, split_alias) or override_args else getattr(self.dataset, split_alias)
        self.root = root if not hasattr(self.dataset, root_alias) or override_args else getattr(self.dataset, root_alias)
        self.base_folder = base_folder if not hasattr(self.dataset, base_folder_alias) or override_args else getattr(self.dataset, base_folder_alias)
        self.indices = indices if not hasattr(self.dataset, indices_alias) or override_args else getattr(self.dataset, indices_alias)

        variations = self._load_t_csv(f"variation_attrs_{num_variations}.txt")
        split_map = {
            "train": [0],
            "valid": [1],
            "test": [2],
            "all": [0,1,2]
        }
        current_split = split_map[self.split]
        ids = [i for i, id in enumerate(variations.split) if id in current_split]

        transitions = list(zip(variations.input, variations.output))
        self.transitions = [(inp, out) for i, (inp, out) in enumerate(transitions) if i in ids]

        self.actions = torch.zeros((len(self.transitions), 2*num_variations))
        for i, id in enumerate(ids):
            id_transition = int(variations.variation[id])
            direction = int(variations.target[id] < variations.source[id] )
            self.actions[i,num_variations*direction+id_transition] = 1.0

    def subset(self, mode: str = "base", limit: Optional[int] = None):
        ld = len(self.dataset)
        lt = len(self.transitions)

        if mode == "base":
            r = range(ld)
        elif mode == "action":
            r = range(ld, ld+lt)
        elif mode == "causal":
            r = range(ld+lt, ld+2*lt)
        
        if limit is not None:
            r = random.sample(r, k=limit)

        return Subset(self,r)
    
    def __getitem__(self, idx : int):
        ld = len(self.dataset)
        lt = len(self.transitions)
        if idx < ld:
            X, target = self.dataset.__getitem__(idx)
            options = {"mode" : "base"}
        else:
            if idx < ld + lt:
                idx = idx - ld
                mode = "action"
            else:
                idx = idx - ld - lt
                mode = "causal"

            x_name, y_name = self.transitions[idx]
            X, target = self.dataset.__getitem__(self.indices.index(x_name))
            Y, _ = self.dataset.__getitem__(self.indices.index(y_name))
            action = self.actions[idx]
            options = {"action": action, "input_y": Y, "mode": mode}
        
        return X, target, options

    def __len__(self) -> int:
        return len(self.dataset) + 2 * len(self.transitions)
     
    def _load_t_csv(self, filename: str) -> T_CSV:
        with open(os.path.join(self.root, self.base_folder, filename)) as csv_file:
            data = list(csv.reader(csv_file))

        headers = data[0]
        data = data[1:]

        inputs = [row[1] for row in data]
        outputs = [row[2] for row in data]
        variations = [row[3] for row in data]
        sources = [int(row[4]) for row in data]
        targets = [int(row[5]) for row in data]
        splits = [int(row[6]) for row in data]

        return T_CSV(inputs, outputs, variations, sources, targets, splits)


class TransitionBatchSampler(Sampler):
    
    def __init__(self, 
        data: TransitionDataset, 
        shuffle: bool, 
        batch_size: int, 
        drop_last: bool, 
        limit: Optional[int] = None, 
        distributed: bool = False
        ) -> None:
        """
        Batch sampler generating batches of a TransitionDataset such that each batch 
        can be of mode ``base``, ``action``, or ``causal``. One batch contains only
        samples with the same mode.

        :param data: (TransitionDataset) Input transition dataset
        :param shuffle: (bool) If true, uses a RandomSampler as sampling strategy, 
                        otherwise, uses SequantialSampler
        :param batch_size: (int) Size of mini-batch.
        :param drop_last: (bool) If ``True``, the sampler will drop the last batch 
                        if its size would be less than ``batch_size``
        :param limit: (Optional[int]) If set, restricts the size of the dataset to 
                        ``limit`` elements drawn randomly
        :param distributed: (bool) If true, creates an instance of DistributedSampler 
                        as part of the sampling strategy
        """
        if shuffle:
            sampler_class = RandomSampler
        else:
            sampler_class = SequentialSampler

        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.limit = limit
        self.distributed = distributed

        self.indices = [
            data.subset("base", limit).indices,
            data.subset("action", limit).indices,
            data.subset("causal", limit).indices
        ]
        self.samplers = [BatchSampler(sampler_class(indices), batch_size, drop_last) for indices in self.indices]
        self.samplers_len = [len(sampler) for sampler in self.samplers]

        self.meta_indices = [0] * self.samplers_len[0] + [1] * self.samplers_len[1] + [2] * self.samplers_len[2]
        
        if distributed:
            self.meta_sampler = DistributedSampler(self.meta_indices, shuffle=shuffle, drop_last=drop_last)
        else:
            self.meta_sampler = sampler_class(self.meta_indices)

    def __iter__(self) -> Iterator[List[int]]:
        meta_sampler_iter = iter(self.meta_sampler)
        samplers_iter = [iter(sampler) for sampler in self.samplers]
        while True:
            try:
                mid = self.meta_indices[next(meta_sampler_iter)]
            except StopIteration: # No indices left
                break

            batch = [self.indices[mid][id] for id in next(samplers_iter[mid])]
            yield batch

    def __len__(self) -> int:
        return len(self.meta_sampler)