import torch
from torch.utils.data import DataLoader

from magneton.io.internal import ProteinDataset

class IDOnlyDataLoader(DataLoader):
    """A DataLoader that only returns the IDs of the proteins in the dataset"""
    def __init__(self, dataset: ProteinDataset, **kwargs):
        super().__init__(dataset, **kwargs)
