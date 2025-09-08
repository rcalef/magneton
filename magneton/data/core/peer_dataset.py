"""
PEER Dataset

This version handles:
1. LMDB datasets (most PEER tasks)
2. CSV datasets (FLIP tasks: AAV, GB1, Thermostability)
3. Different archive formats (tar.gz, zip)
4. Different directory structures
"""

import os
import ssl
import tarfile
import zipfile
import urllib.request
import pandas as pd
import lmdb
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional
from torch.utils.data import Dataset
from tqdm import tqdm

from .core_dataset import DataElement

# Handle SSL certificate issues
ssl._create_default_https_context = ssl._create_unverified_context

# Dataset configurations with format information
PEER_DATASET_INFO = {
    # LMDB-based datasets
    "fold": {
        "url": "http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/remote_homology.tar.gz",
        "format": "lmdb",
        "archive_type": "tar.gz",
        "extracted_dir": "remote_homology",
        "lmdb_pattern": "remote_homology/remote_homology_{split}.lmdb",
        "task_type": "multiclass",
    },
    
    "fluorescence": {
        "url": "http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/fluorescence.tar.gz",
        "format": "lmdb",
        "archive_type": "tar.gz", 
        "extracted_dir": "fluorescence",
        "lmdb_pattern": "fluorescence/fluorescence_{split}.lmdb",
        "task_type": "regression",
        "target_field": "log_fluorescence"
    },
    
    "stability": {
        "url": "http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/stability.tar.gz",
        "format": "lmdb",
        "archive_type": "tar.gz",
        "extracted_dir": "stability",
        "lmdb_pattern": "stability/stability_{split}.lmdb",
        "task_type": "regression",
        "target_field": "stability_score"
    },
    
    "beta_lactamase": {
        "url": "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/peerdata/beta_lactamase.tar.gz",
        "format": "lmdb",
        "archive_type": "tar.gz",
        "extracted_dir": "beta_lactamase",
        "lmdb_pattern": "beta_lactamase/beta_lactamase_{split}.lmdb",
        "task_type": "regression",
        "target_field": "scaled_effect1",
    },
    
    "solubility": {
        "url": "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/peerdata/solubility.tar.gz",
        "format": "lmdb",
        "archive_type": "tar.gz",
        "extracted_dir": "solubility",
        "lmdb_pattern": "solubility/solubility_{split}.lmdb",
        "task_type": "binary",
    },
    
    "subcellular_localization": {
        "url": "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/peerdata/subcellular_localization.tar.gz",
        "format": "lmdb",
        "archive_type": "tar.gz",
        "extracted_dir": "subcellular_localization",
        "lmdb_pattern": "subcellular_localization/subcellular_localization_{split}.lmdb",
        "task_type": "binary",
        "target_field": "localization",
    },
    
    "binary_localization": {
        "url": "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/peerdata/subcellular_localization_2.tar.gz",
        "format": "lmdb",
        "archive_type": "tar.gz",
        "extracted_dir": "subcellular_localization_2",
        "lmdb_pattern": "subcellular_localization_2/subcellular_localization_2_{split}.lmdb",
        "task_type": "binary",
        "target_field": "localization",
    },
    
    # FLIP CSV-based datasets
    "aav": {
        "url": "https://github.com/J-SNACKKB/FLIP/raw/d5c35cc716ca93c3c74a0b43eef5b60cbf88521f/splits/aav/splits.zip",
        "format": "csv",
        "archive_type": "zip",
        "extracted_dir": "splits",
        "csv_files": {
            "train": "splits/two_vs_many.csv",
            "valid": "splits/two_vs_many.csv",
            "val": "splits/two_vs_many.csv",  # Lightning uses "val"
            "test": "splits/two_vs_many.csv",
        },
        "task_type": "regression",
        "sequence_col": "sequence",
        "target_col": "target",
    },
    
    "gb1": {
        "url": "https://github.com/J-SNACKKB/FLIP/raw/d5c35cc716ca93c3c74a0b43eef5b60cbf88521f/splits/gb1/splits.zip",
        "format": "csv",
        "archive_type": "zip",
        "extracted_dir": "splits",
        "csv_files": {
            "train": "splits/two_vs_rest.csv",
            "valid": "splits/two_vs_rest.csv",
            "val": "splits/two_vs_rest.csv",  # Lightning uses "val"
            "test": "splits/two_vs_rest.csv",
        },
        "task_type": "regression",
        "sequence_col": "sequence",
        "target_col": "target",
    },
    
    "thermostability": {
        "url": "https://github.com/J-SNACKKB/FLIP/raw/d5c35cc716ca93c3c74a0b43eef5b60cbf88521f/splits/meltome/splits.zip",
        "format": "csv",
        "archive_type": "zip",
        "extracted_dir": "splits",
        "csv_files": {
            "train": "splits/human_cell.csv",
            "valid": "splits/human_cell.csv",
            "val": "splits/human_cell.csv",  # Lightning uses "val"
            "test": "splits/human_cell.csv",
        },
        "task_type": "regression",
        "sequence_col": "sequence",
        "target_col": "target",
    },
}


class PeerDataset(Dataset):
    """PEER dataset handler that supports both LMDB and CSV formats"""
    
    def __init__(self, task: str, data_dir: str, split: str = "train"):
        self.task = task
        self.split = split
        self.data_dir = Path(data_dir)
        
        if task not in PEER_DATASET_INFO:
            raise ValueError(f"Unknown task: {task}")
        
        self.info = PEER_DATASET_INFO[task]
        
        # Try to find existing directory with different cases
        possible_dirs = [
            self.data_dir / task.lower(),
        ]
        
        self.task_dir = None
        for pdir in possible_dirs:
            if pdir.exists():
                self.task_dir = pdir
                break
        
        # If no existing directory found, create with lowercase
        if self.task_dir is None:
            self.task_dir = self.data_dir / task.lower()
            self.task_dir.mkdir(parents=True, exist_ok=True)
        
        # Download and extract if needed
        self._ensure_data_available()
        
        # Load the dataset
        if self.info["format"] == "lmdb":
            self.data = self._load_lmdb_data()
        elif self.info["format"] == "csv":
            self.data = self._load_csv_data()
        else:
            raise ValueError(f"Unknown format: {self.info['format']}")
    
    def _ensure_data_available(self):
        """Download and extract dataset if needed"""
        # Check if data already extracted
        extracted_dir = self.task_dir / self.info["extracted_dir"]
        if extracted_dir.exists():
            # Quick check if it has expected files
            if self.info["format"] == "lmdb":
                # Check for at least one LMDB file
                pattern = self.info["lmdb_pattern"].replace("{split}", "*")
                if list(self.task_dir.glob(pattern)):
                    return
            elif self.info["format"] == "csv":
                # Check for CSV files
                csv_file = self.task_dir / list(self.info["csv_files"].values())[0]
                if csv_file.exists():
                    return
        
        # Download if needed
        url = self.info["url"]
        archive_name = url.split("/")[-1]
        archive_path = self.task_dir / archive_name
        
        if not archive_path.exists():
            print(f"Downloading {self.task} from {url}...")
            urllib.request.urlretrieve(url, archive_path)
        
        # Extract
        print(f"Extracting {archive_name}...")
        if self.info["archive_type"] == "zip":
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(self.task_dir)
        elif self.info["archive_type"] == "tar.gz":
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(self.task_dir)
        else:
            raise ValueError(f"Unknown archive type: {self.info['archive_type']}")
    
    def _load_lmdb_data(self) -> List[Dict]:
        """Load data from LMDB file"""
        # Map split names
        split_map = {"val": "valid"}
        actual_split = split_map.get(self.split, self.split)
        
        # Get LMDB path
        lmdb_pattern = self.info["lmdb_pattern"]
        lmdb_path = self.task_dir / lmdb_pattern.format(split=actual_split)
        
        if not lmdb_path.exists():
            raise FileNotFoundError(f"LMDB not found: {lmdb_path}")
        
        data = []
        with lmdb.open(str(lmdb_path), readonly=True, lock=False) as env:
            with env.begin() as txn:
                for key, value in txn.cursor():
                    try:
                        item = pickle.loads(value)
                        # Skip metadata entries that aren't protein data
                        if not isinstance(item, dict):
                            continue
                        item['_key'] = key.decode()
                        data.append(item)
                    except Exception as e:
                        print(f"Error loading item {key}: {e}")
        
        return data
    
    def _load_csv_data(self) -> List[Dict]:
        """Load data from CSV file"""
        csv_file_path = self.info["csv_files"][self.split]
        csv_file = self.task_dir / csv_file_path
        
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV not found: {csv_file}")
        
        df = pd.read_csv(csv_file)
        
        # For FLIP datasets, we need to filter by split column
        split_column = None
        if "split" in df.columns:
            split_column = "split"
        elif "set" in df.columns:
            split_column = "set"
            
        if split_column is not None:
            # Map our split names to FLIP split values
            split_mapping = {
                "train": "train",
                "val": "val",
                "valid": "val",
                "test": "test"
            }
            flip_split = split_mapping.get(self.split, self.split)
            df = df[df[split_column] == flip_split]
            
            if len(df) == 0:
                raise ValueError(f"No data found for split '{flip_split}' in {csv_file}")
        
        data = []
        seq_col = self.info["sequence_col"]
        target_col = self.info["target_col"]
        
        for idx, row in df.iterrows():
            item = {
                "primary": row[seq_col],
                target_col: row[target_col],
                "_key": str(idx)
            }
            data.append(item)
        
        return data
    
    def _convert_to_data_element(self, item: Dict) -> DataElement:
        """Convert raw data to appropriate DataElement type"""
        task_type = self.info["task_type"]
        
        # Extract sequence
        seq = item.get("primary", item.get("sequence", ""))
        if isinstance(seq, bytes):
            seq = seq.decode('utf-8')
        
        # Extract target
        target_field = self.info.get("target_field", "target")
        if task_type == "multiclass" and self.task == "fold":
            target_field = "fold_label"
        elif task_type == "regression" and self.task == "fluorescence":
            target_field = "log_fluorescence"
        elif task_type == "regression" and self.task == "stability":
            target_field = "stability_score"
        elif self.task == "solubility":
            target_field = "solubility"
        elif self.task == "subcellular_localization" or self.task == "binary_localization":
            target_field = "localization"
        
        labels = item.get(target_field)
        
        # Convert labels to tensor
        if labels is not None:
            if task_type == "multiclass":
                labels = torch.tensor(labels)
            elif task_type in ["binary", "regression"]:
                labels = torch.tensor([labels], dtype=torch.float)
            elif task_type == "multilabel":
                labels = torch.tensor(labels, dtype=torch.float)
            else:
                # Default: wrap in tensor
                labels = torch.tensor(labels)
        
        return DataElement(
            protein_id=item['_key'],
            length=len(seq),
            seq=seq,
            labels=labels
        )
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self._convert_to_data_element(self.data[idx])


# For backward compatibility
PEER_TASK_TO_CONFIGS = {
    task: {"task_type": info["task_type"]} 
    for task, info in PEER_DATASET_INFO.items()
}

class PeerDataModule:
    """DataModule wrapper"""
    
    def __init__(self, task_name: str, data_dir: Path):
        self.task_name = task_name
        self.data_dir = data_dir
        self.info = PEER_DATASET_INFO.get(task_name, {})
        
    def get_dataset(self, split: str) -> PeerDataset:
        return PeerDataset(self.task_name, self.data_dir, split)
    
    def num_classes(self) -> int:
        """Get number of classes"""
        info = self.info
        if "num_classes" in info:
            return info["num_classes"]
        elif info.get("task_type") == "binary":
            return 1  # BCEWithLogitsLoss expects 1 output for binary classification
        elif info.get("task_type") == "regression":
            return 1
        elif info.get("task_type") == "multiclass":
            # Need to determine from data
            if self.task_name == "fold":
                return 1195
            elif self.task_name == "subcellular_localization":
                return 10
        return 1