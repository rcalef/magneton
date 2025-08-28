import os

from dataclasses import dataclass, replace
from functools import partial
from typing import Callable, List, Tuple

import torch

from esm.tokenization import get_esmc_model_tokenizers
from esm.utils.misc import stack_variable_length_tensors
from torchdata.nodes import BaseNode, ParallelMapper

from magneton.config import DataConfig, TrainingConfig
from ..core import Batch, DataElement
from magneton.embedders.base_embedder import BaseDataModule

@dataclass(kw_only=True)
class ESMCDataElement(DataElement):
    """Single data element for ESM-C.

    - protein_id (str): UniProt ID of protein.
    - tokenized_seq (torch.Tensor): Tokenized AA seq.
    - substructures (List[LabeledSubstructure]): List of annotated substructures for this protein.
    """
    tokenized_seq: torch.Tensor


@dataclass(kw_only=True)
class ESMCBatch(Batch):
    tokenized_seq: torch.Tensor

    def to(self, device: str):
        super().to(device)
        self.tokenized_seq = self.tokenized_seq.to(device)
        return self


class ESMCTransformNode(ParallelMapper):
    def __init__(
        self,
        source_node: BaseNode,
        num_workers: int = 2,
    ):
        tokenizer = get_esmc_model_tokenizers()
        def _process(
            x: DataElement,
        ) -> ESMCDataElement:
            return ESMCDataElement(
                protein_id=x.protein_id,
                length=x.length,
                tokenized_seq=torch.tensor(tokenizer.encode(x.seq)),
                substructures=x.substructures,
                labels=x.labels,
            )
        super().__init__(source=source_node, map_fn=_process, num_workers=num_workers)


def get_esmc_collate_fn() -> Callable:
    tokenizer = get_esmc_model_tokenizers()
    return partial(esmc_collate, pad_id=tokenizer.pad_token_id)


def esmc_collate(
    entries: List[ESMCDataElement],
    pad_id: int,
) -> ESMCBatch:
    """
    Collate the entries into a batch.
    """
    protein_ids = [x.protein_id for x in entries]
    padded_tensor = stack_variable_length_tensors(
        [x.tokenized_seq for x in entries],
        constant_value=pad_id,
    )
    # Each of these should either all be set or all be None
    if entries[0].substructures is None:
        substructs = None
    else:
        substructs = [x.substructures for x in entries]
    if entries[0].labels is None:
        labels = None
    else:
        labels = torch.stack([x.labels for x in entries])
    return ESMCBatch(
        protein_ids=protein_ids,
        tokenized_seq=padded_tensor,
        substructures=substructs,
        labels=labels,
    )


class ESMCDataModule(BaseDataModule):
    def __init__(
        self,
        data_config: DataConfig,
        train_config: TrainingConfig,
    ):
        super().__init__(data_config, train_config)

    def _get_split_info(self, split: str) -> Tuple[str, str]:
        if split == "all":
            return self.data_config.data_dir, self.data_config.prefix
        else:
            return (
                os.path.join(self.data_config.data_dir, f"{split}_sharded"),
                f"swissprot.with_ss.{split}",
            )

    def _get_dataloader(
        self,
        split: str,
        **kwargs,
    ) -> torch.utils.data.DataLoader:
        data_dir, prefix = self._get_split_info(split)
        config = replace(
            self.data_config,
            data_dir=data_dir,
            prefix=prefix,
        )
        dataset = ESMCDataSet(
            data_config=config,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            collate_fn=partial(esmc_collate, pad_id=dataset.tokenizer.pad_token_id),
            num_workers=3,
            **kwargs,
        )

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
