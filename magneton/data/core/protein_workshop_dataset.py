import ssl

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import lightning as L
import numpy as np
import proteinworkshop
import torch

from graphein.protein import AMINO_ACIDS
from proteinworkshop.datasets.base import ProteinDataset
from torch.utils.data import (
    Dataset,
)
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from tqdm import tqdm

from .core_dataset import DataElement

ssl._create_default_https_context = ssl._create_unverified_context
torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage])

CONFIG_PATH = (
    Path(proteinworkshop.__path__[0]) /
    "config"
)

TASK_TO_CONFIGS = {
    "ppi": {"dataset": "masif_site", "task": "ppi_site_prediction"},
    "go-bp": {"dataset": "go-bp", "task": "multilabel_graph_classification"},
    "go-cc": {"dataset": "go-cc", "task": "multilabel_graph_classification"},
    "go-mf": {"dataset": "go-mf", "task": "multilabel_graph_classification"},
    "EC": {"dataset": "ec_reaction", "task": "multilabel_graph_classification"},
}

TASK_TO_SUBDIR = {
    "ppi": "masif_ppi",
    "go-bp": "GeneOntology",
    "go-cc": "GeneOntology",
    "go-mf": "GeneOntology",
    "EC": "EnzymeCommission",
}

AA_LOOKUP = np.array(AMINO_ACIDS)

class WorkshopDataset(Dataset):
    def __init__(
        self,
        source_dataset: ProteinDataset,
    ):
        self.source_dataset = source_dataset
        self.pdb_dir = Path(source_dataset.pdb_dir)
        self.format = source_dataset.format

        # We don't actually want the graphs for now, so just read through the dataset once to
        # get the parts we care about. Can change this in the future if we use a model
        # that would actually be able to directly use their graphs.
        print("converting ProteinWorkshop dataset")
        skipped = 0
        tot = 0
        converted_items = []
        for item in tqdm(self.source_dataset):
            tot += 1
            if item is None:
                skipped += 1
                continue
            converted_items.append(self._convert_item(item))
        print(f"skipped {skipped} / {tot}")
        self.dataset = converted_items

    def _convert_item(self, item: Data) -> DataElement:
        # Extract just PDB ID if chain IDs are being appended
        pdb_id = item.id
        if self.source_dataset.chains is not None:
            pdb_id = item.id.split("_")[0]

        # Get sequence as string
        seq = "".join(AA_LOOKUP[item.residue_type].tolist())

        # Get full path to structure file
        if self.format == "pdb":
            struct_path = self.pdb_dir / f"{pdb_id}.{self.format}"
        else:
            struct_path = self.pdb_dir / f"{pdb_id}.{self.format}.gz"

        # Get labels if present
        if hasattr(item, "node_y"):
            labels = item.node_y
        else:
            labels = item.graph_y

        return DataElement(
            protein_id=pdb_id,
            length=len(seq),
            labels=labels.squeeze().int(),
            seq=seq,
            structure_path=str(struct_path),
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index]


class WorkshopDataModule:
    def __init__(
        self,
        task_name: str,
        data_dir: Path,
    ):
        task_data = TASK_TO_CONFIGS[task_name]
        overrides = [f"{k}={v}" for k,v in task_data.items()]
        subdir = TASK_TO_SUBDIR[task_name]

        data_overrides = {
            "path": data_dir / subdir,
            "pdb_dir": data_dir / "workshop_pdbs",
            "in_memory": False,
            "format": "pdb",
        }

        overrides.extend([f"++dataset.datamodule.{k}={v}" for k,v in data_overrides.items()])

        hydra.core.global_hydra.GlobalHydra.instance().clear()
        with hydra.initialize_config_dir(config_dir=str(CONFIG_PATH)):
            cfg = hydra.compose(config_name="finetune", overrides=overrides)

        self.workshop_module = hydra.utils.instantiate(cfg.dataset.datamodule)
        #self.workshop_module.setup()
        self._num_classes = cfg.dataset.num_classes

    def get_dataset(
        self,
        split: str,
    ) -> WorkshopDataset:
        if split == "train":
            source_dataset = self.workshop_module.train_dataset()
        elif split == "val":
            source_dataset = self.workshop_module.val_dataset()
        elif split == "test":
            source_dataset = self.workshop_module.test_dataset()
        else:
            raise ValueError(f"unknown split: {split}")
        return WorkshopDataset(source_dataset=source_dataset)

    def num_classes(self) -> int:
        return self._num_classes

