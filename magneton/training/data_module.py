from typing import Optional
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from ..config.config import DataConfig

# TODO
# Incorporate ProteinDataset and splits into here
# Refactor to have that the splits are made first and then embedded

class ProteinDataModule(pl.LightningDataModule):
    def __init__(self, config: DataConfig):
        super().__init__()
        self.config = config
        self.data_dir = config.data_dir
        
    def setup(self, stage: Optional[str] = None):
        # Load embeddings and labels
        embeddings = np.load(f"{self.data_dir}/{self.config.embedding_type}_embeddings.npy")
        labels = np.load(f"{self.data_dir}/{self.config.embedding_type}_labels.npy")
        
        # Convert to tensors
        embeddings = torch.FloatTensor(embeddings)
        labels = torch.LongTensor(labels)
        
        # Create dataset
        dataset = TensorDataset(embeddings, labels)
        
        # Split dataset
        total_size = len(dataset)
        train_size = int(self.config.train_split * total_size)
        val_size = int(self.config.val_split * total_size)
        test_size = total_size - train_size - val_size
        
        self.train_dataset, self.val_dataset, self.test_dataset = \
            random_split(dataset, [train_size, val_size, test_size])
            
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers
        ) 