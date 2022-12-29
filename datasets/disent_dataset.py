
import os

from torch.utils.data import Dataset, Subset
from typing import Callable

from disent.dataset import DisentDataset
from disent.dataset.data import Cars3dData, DSpritesData, DSpritesImagenetData, Mpi3dData, SmallNorbData, Shapes3dData, SpritesData, XYSquaresData, XYObjectData, XYObjectShadedData

from .transition import TransitionDataset

from collections import namedtuple
import csv

PARTITION = namedtuple("PARTITION", ["index", "split"])

class DisentLibDataset(Dataset):
    DATASETS = {
        "cars3d": Cars3dData,
        "dsprites": DSpritesData,
        # "dSpritesImageNet": DSpritesImagenetData,
        # "mpi3d": Mpi3dData,
        "smallnorb": SmallNorbData,
        "3dshapes": Shapes3dData,
        "sprites": SpritesData,
        # "XYSquares": XYSquaresData,
        # "XYObject": XYObjectData,
        # "XYObjectShaded": XYObjectShadedData
    }

    def __init__(self, 
                data_dir: str,
                dataset_name: str, 
                split: str,
                transform: Callable,
                **kwargs):

        self.split = split
        self.root = data_dir
        self.base_folder = dataset_name

        data = DisentDataset(DisentLibDataset.DATASETS[dataset_name](data_root=data_dir,prepare=True,transform=transform), return_factors=True)
        self._full_data = data

        if split == "all":
            self.indices = list(map(str, splits.index))
            self.data = data
        else:
            split_map = {
                "train": 0,
                "valid": 1,
                "test": 2,
            }
            split_ = split_map[split]
            splits = self._load_csv("list_eval_partition.txt")
            indices = [i for i in splits.index if splits.split[i] == split_]

            self.indices = list(map(str, indices))
            self.data = Subset(data, indices)

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_idx = self.data[idx]
        return data_idx['x_targ'][0], data_idx['factors'][0]

    
    def _load_csv(self, filename: str) -> PARTITION:
        with open(os.path.join(self.root, self.base_folder, filename)) as csv_file:
            data = list(csv.reader(csv_file))

        headers = data[0]
        data = data[1:]

        indexes = [int(row[1]) for row in data]
        splits = [int(row[2]) for row in data]

        return PARTITION(indexes, splits)



def MyCars3D(*args, **kwargs):
    return DisentLibDataset(*args, dataset_name="cars3d", **kwargs)

def MyDSprites(*args, **kwargs):
    return DisentLibDataset(*args, dataset_name="dsprites", **kwargs)

def MySmallNORB(*args, **kwargs):
    return DisentLibDataset(*args, dataset_name="smallnorb", **kwargs)

def MyShapes3D(*args, **kwargs):
    return DisentLibDataset(*args, dataset_name="3dshapes", **kwargs)

def MySprites(*args, **kwargs):
    return DisentLibDataset(*args, dataset_name="sprites", **kwargs)


def TCars3D(*args, **kwargs):
    return TransitionDataset(DisentLibDataset(*args, dataset_name="cars3d", **kwargs), num_variations=3) # ('elevation', 'azimuth', 'object_type')

def TDSprites(*args, **kwargs):
    return TransitionDataset(DisentLibDataset(*args, dataset_name="dsprites", **kwargs), num_variations=5) # ('shape', 'scale', 'orientation', 'position_x', 'position_y')

def TSmallNORB(*args, **kwargs):
    return TransitionDataset(DisentLibDataset(*args, dataset_name="smallnorb", **kwargs), num_variations=5) # ('category', 'instance', 'elevation', 'rotation', 'lighting')

def TShapes3D(*args, **kwargs):
    return TransitionDataset(DisentLibDataset(*args, dataset_name="3dshapes", **kwargs), num_variations=6) # ('floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation')

def TSprites(*args, **kwargs):
    return TransitionDataset(DisentLibDataset(*args, dataset_name="sprites", **kwargs), num_variations=9) # ('bottomwear', 'topwear', 'hair', 'eyes', 'shoes', 'body', 'action', 'rotation', 'frame')




