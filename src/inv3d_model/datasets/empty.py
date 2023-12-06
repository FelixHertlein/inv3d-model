from typing import *

from torch.utils.data import Dataset

from .loaders import *


class EmptyDataset(Dataset):
    def __init__(self, *args, **kwargs):
        pass

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {}
