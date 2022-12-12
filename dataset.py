import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
import csv
from collections import namedtuple


# Add your custom dataset class here
class MyDataset(Dataset):
    def __init__(self):
        pass
    
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass


class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """
    
    def _check_integrity(self) -> bool:
        return True
        
T_CSV = namedtuple("T_CSV", ["input", "output", "variation", "source", "target", "split"])
class TransitionCelebA(CelebA):
    """
    CelebA dataset with transitions between images with similar features.
    """

    def __init__(self, *args, num_variations = 10, **kwargs):
        super(TransitionCelebA, self).__init__(*args, **kwargs)

        variations = self._load_t_csv("variation_attrs.txt")
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
            id_transition = self.attr_names.index(variations.variation[id])
            direction = int(variations.target[id] < 0)
            self.actions[i,num_variations*direction+id_transition] = 1.0

    def subset(self, mode: str = "base"):
        la = len(self.attr)
        lt = len(self.transitions)

        if mode == "base":
            r = range(la)
        elif mode == "action":
            r = range(la, la+lt)
        elif mode == "causal":
            r = range(la+lt, la+2*lt)

        return torch.utils.data.Subset(self,r)
    
    def __getitem__(self, idx):
        if idx < len(self.attr):
            X, target = super(TransitionCelebA,self).__getitem__(idx)
            options = {"mode" : "base"}
        else:
            if idx < len(self.attr) + len(self.transitions):
                idx = idx - len(self.attr)
                mode = "action"
            else:
                idx = idx - len(self.attr) - len(self.transitions)
                mode = "causal"
            x_name, y_name = self.transitions[idx]
            X, target = super(TransitionCelebA,self).__getitem__(self.filename.index(x_name))
            Y, _ = super(TransitionCelebA,self).__getitem__(self.filename.index(y_name))
            action = self.actions[idx]
            options = {"action": action, "input_y": Y, "mode": mode}
        
        return X, target, options

    def __len__(self) -> int:
        return len(self.attr) + 2 * len(self.transitions)
     
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
    
    def _check_integrity(self) -> bool:
        return True
    
    

class OxfordPets(Dataset):
    """
    URL = https://www.robots.ox.ac.uk/~vgg/data/pets/
    """
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 transform: Callable,
                **kwargs):
        self.data_dir = Path(data_path) / "OxfordPets"        
        self.transforms = transform
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])
        
        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, 0.0 # dummy data to prevent breaking 

class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        mode: str = "base",
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.mode = mode

    def setup(self, stage: Optional[str] = None) -> None:   
#       =========================  CelebA Dataset  =========================
    
        train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.CenterCrop(148),
                                              transforms.Resize(self.patch_size),
                                              transforms.ToTensor(),])
        
        val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(self.patch_size),
                                            transforms.ToTensor(),])
        
        # self.train_dataset = MyCelebA(
        self.train_dataset = TransitionCelebA(
            self.data_dir,
            split='train',
            transform=train_transforms,
            download=False,
        ).subset(self.mode)
        
        # Replace CelebA with your dataset
        # self.val_dataset = MyCelebA(
        self.val_dataset = TransitionCelebA(
            self.data_dir,
            split='test',
            transform=val_transforms,
            download=False,
        )  .subset(self.mode)   
#       ===============================================================
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,#144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
     