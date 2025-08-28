from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Literal

import lightning as L
import pandas as pd
import torch

from pysam import FastaFile
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from .core_dataset import DataElement

class SupervisedDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        num_classes: int,
    ):
        self.df = df
        self.num_classes = num_classes

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        # Sum first dim because of multilabel classification, i.e.
        # if this item has N labels (e.g. N GO terms), this does:
        # (N X num_classes) -> (num_classes)
        one_hot_labels = torch.nn.functional.one_hot(
            row.labels,
            num_classes=self.num_classes,
        ).sum(dim=0)
        return DataElement(
            protein_id=row.protein_id,
            labels=one_hot_labels,
            seq=row.seq,
            structure_path=row.structure_path,
        )

LABEL_LINE: Dict[str, int] = {
    "MF": 1,
    "BP": 5,
    "CC": 9,
    "EC": 1,
}

@dataclass
class DeepFRIDataConfig:
    data_dir: str
    task: Literal["BP", "CC", "MF", "EC"]
    batch_size: int = 32

class DeepFRIDataModule(L.LightningDataModule):
    """Data module encompassing evaluation sets from DeepFRI (GO terms and EC numbers)

    Structure largely adapted from ProteinWorkshop.
    """
    def __init__(
        self,
        config: DeepFRIDataConfig,
        seq_tokenizer: PreTrainedTokenizerBase | None,
    ):
        super().__init__()
        self.task = config.task
        self.batch_size = config.batch_size
        self.seq_tokenizer = seq_tokenizer

        data_dir = Path(config.data_dir)
        if self.task == "EC":
            data_dir = data_dir / "EnzymeCommission"

            self.label_fname = data_dir / "nrPDB-EC_2020.04_annot.tsv"
            self.train_fname = data_dir / "nrPDB-EC_2020.04_train.txt"
            self.val_fname = data_dir / "nrPDB-EC_2020.04_valid.txt"
            self.test_fname = data_dir / "nrPDB-EC_2020.04_test.txt"
            fa_path = data_dir / "nrPDB-EC_2020.04_sequences.fasta"

            n_header_rows = 3
            col_names = ["PDB", "EC"]
        else:
            data_dir = data_dir / "GeneOntology"

            self.label_fname = data_dir / "nrPDB-GO_annot.tsv"
            self.train_fname = data_dir / "nrPDB-GO_train.txt"
            self.val_fname = data_dir / "nrPDB-GO_valid.txt"
            self.test_fname = data_dir / "nrPDB-GO_test.txt"
            fa_path = data_dir / "nrPDB-GO_sequences.fasta"

            n_header_rows = 13
            col_names = ["PDB", "MF", "BP", "CC"]

        # Parse labels
        label_line = LABEL_LINE[self.task]
        with open(self.label_fname, "r") as f:
            all_labels = f.readlines()[label_line].strip("\n").split("\t")
        self.labeller = LabelEncoder().fit(all_labels)
        self.num_classes = len(self.labeller.classes_)

        # Load labels for full dataset
        df = (
            pd.read_table(
                self.label_fname,
                skiprows=n_header_rows,
                names=col_names,
            )
            .loc[lambda x: ~x[self.task].isna()]
            .assign(
                labels=lambda x: x[self.task].str.split(",").map(self.labeller.transform),
                labels_tens=lambda x: x.labels.map(torch.tensor),
            )
            .drop(columns=["labels"])
            .rename(columns={"PDB": "protein_id", "labels_tens": "labels"})
        )
        self.dataset = df[["protein_id", self.task, "labels"]]

        self.fa = FastaFile(fa_path)
        self.all_seq_ids = set(self.fa.references)

    def parse_dataset(
        self,
        split: Literal[
            "training",
            "validation",
            "testing"
        ],
    ) -> pd.DataFrame:
        """
        Parses the raw dataset files to Pandas DataFrames.
        Maps classes to numerical values.
        """
        # Read in IDs of structures in split
        if split == "training":
            data = pd.read_csv(self.train_fname, sep="\t", header=None)[0]
        elif split == "validation":
            data = pd.read_csv(self.val_fname, sep="\t", header=None)[0]
        elif split == "testing":
            data = pd.read_csv(self.test_fname, sep="\t", header=None)[0]
        else:
            raise ValueError(f"Unknown split: {split}")

        want_subset = (
            self.dataset
            .loc[lambda x: x.protein_id.isin(data)]
            .loc[lambda x: x.protein_id.isin(self.all_seq_ids)]
            .assign(
                seq=lambda x: x.protein_id.map(self.fa.fetch),
                structure_path="NA",
            )
        )
        return want_subset


    def _get_dataloader(
        self,
        split: Literal[
            "training",
            "validation",
            "testing"
        ],
        **kwargs,
    ) -> torch.utils.data.DataLoader:
        df = self.parse_dataset(split)
        dataset = SupervisedDataset(df=df, num_classes=self.num_classes)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=partial(supervised_collate, tokenizer=self.seq_tokenizer),
            num_workers=3,
            **kwargs,
        )

    def train_dataloader(self):
        return self._get_dataloader(
            "training",
            shuffle=True,
        )

    def val_dataloader(self):
        return self._get_dataloader(
            "validation",
            shuffle=False,
        )

    def test_dataloader(self):
        return self._get_dataloader(
            "testing",
            shuffle=False,
        )