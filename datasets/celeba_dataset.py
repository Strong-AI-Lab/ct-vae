
from torchvision.datasets import CelebA

from .transition import TransitionDataset

from typing import Union


class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """
    
    # def __getitem__(self, idx: Union[int, str]):
    #     if type(idx) == int:
    #         return super(MyCelebA, self).__getitem__(idx) # id is given
    #     return super(MyCelebA, self).__getitem__(self.filename.find(idx)) # image name is given
    
    def _check_integrity(self) -> bool:
        return True


def TCeleba(*args, **kwargs):
    return TransitionDataset(MyCelebA(*args, **kwargs), num_variations= 10, indices_alias="filename")

