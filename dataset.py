
from typing import List, Optional, Sequence, Union
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from datasets.celeba_dataset import MyCelebA, TCeleba
from datasets.disent_dataset import MyCars3D, MyDSprites, MySmallNORB, MyShapes3D, MySprites, TCars3D, TDSprites, TSmallNORB, TShapes3D, TSprites
from datasets.oxford_dataset import OxfordPets



DATASETS = {
    "Celeba" : MyCelebA,
    "TCeleba": TCeleba,
    "Cars3D" : MyCars3D,
    "TCars3D": TCars3D,
    "DSprites" : MyDSprites,
    "TDSprites": TDSprites,
    "SmallNORB" : MySmallNORB,
    "TSmallNORB": TSmallNORB,
    "Shapes3D" : MyShapes3D,
    "TShapes3D": TShapes3D,
    "Sprites" : MySprites,
    "TSprites": TSprites,
}


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
        dataset_name: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        mode: str = "base",
        limit: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.dataset_name = dataset_name
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.mode = mode
        self.limit = limit
    
    def get_mode(self, it):
        if type(self.mode) is str:
            return self.mode
        if type(self.mode) is list:
            return self.mode[it % len(self.mode)]
        raise TypeError("Mode has incorrect type")

    def setup(self, stage: Optional[str] = None) -> None:   
#       =========================  CelebA Dataset  =========================
        self.iteration = 0
    
        train_transforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.CenterCrop(148),
                                              transforms.Resize(self.patch_size)])
        
        val_transforms = transforms.Compose([transforms.ToTensor(),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(self.patch_size),])
                                            
        self.train_dataset = DATASETS[self.dataset_name](
            self.data_dir,
            split='train',
            transform=train_transforms,
            download=False,
        )
        
        self.val_dataset = DATASETS[self.dataset_name](
            self.data_dir,
            split='test',
            transform=val_transforms,
            download=False,
        )

        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset.subset(self.get_mode(self.iteration), self.limit),
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset.subset(self.get_mode(self.iteration)),
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        self.iteration += 1
        return DataLoader(
            self.val_dataset.subset(self.get_mode(self.iteration - 1)),
            batch_size=self.val_batch_size,#144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
     