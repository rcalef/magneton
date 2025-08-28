import os

from dataclasses import dataclass, replace
from functools import partial
from typing import List, Tuple

import torch
import lightning as L

from esm.tokenization import get_esmc_model_tokenizers
from esm.utils.misc import stack_variable_length_tensors
from torchdata.nodes import Batcher, Loader, Mapper

from magneton.config import DataConfig
from magneton.types import DataType

from .core import get_core_node
from .model_specific import (
    ESMCTransformNode,
    get_esmc_collate_fn,
)

model_data = {
    "esmc": (ESMCTransformNode, get_esmc_collate_fn(), [DataType.SEQ]),
}

class MagnetonDataModule(L.LightningDataModule):
    """DataModule for substructure-aware fine-tuning."""
    def __init__(
        self,
        data_config: DataConfig,
        model_type: str,
        distributed: bool = False,
    ):
        self.data_config = data_config
        transform_cls, collate_fn, want_datatypes = model_data[model_type]
        self.transform_cls = transform_cls
        self.collate_fn = collate_fn
        self.want_datatypes = want_datatypes + [DataType.SUBSTRUCT]
        self.distributed = distributed

    def _get_dataloader(
        self,
        split: str,
        **kwargs,
    ) -> Loader:
        node = get_core_node(
            data_config=self.data_config,
            want_datatypes=self.want_datatypes,
            split=split,
            distributed=self.distributed,
            **kwargs,
        )
        node = self.transform_cls(node)
        node = Batcher(node, batch_size=self.data_config.batch_size)
        node = Mapper(node, self.collate_fn)

        return Loader(node)

    def train_dataloader(self):
        return self._get_dataloader(
            "train",
            shuffle=True,
        )

    def val_dataloader(self):
        return self._get_dataloader(
            "val",
            shuffle=False,
        )

    def test_dataloader(self):
        return self._get_dataloader(
            "test",
            shuffle=False,
        )

    def predict_dataloader(self):
        return self._get_dataloader(
            "all",
            shuffle=False,
        )
