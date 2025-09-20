from pathlib import Path

from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal

import torch

from esm.utils.misc import stack_variable_length_tensors
from torchdata.nodes import BaseNode, ParallelMapper
from transformers import EsmTokenizer

from ..core import Batch, DataElement

@dataclass(kw_only=True)
class ESM2DataElement(DataElement):
    """Single data element for ESM-C.

    - tokenized_seq (torch.Tensor): Tokenized AA seq.
    """
    tokenized_seq: torch.Tensor


@dataclass(kw_only=True)
class ESM2Batch(Batch):
    tokenized_seq: torch.Tensor

    def to(self, device: str):
        super().to(device)
        self.tokenized_seq = self.tokenized_seq.to(device)
        return self


class ESM2TransformNode(ParallelMapper):
    def __init__(
        self,
        source_node: BaseNode,
        data_dir: str | Path = None,
        tokenizer_path: str
        | Path = "/home/rcalef/storage/om_storage/model_weights/esm2_t33_650M_UR50D",
        num_workers: int = 2,
    ):
        self.tokenizer = EsmTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )
        def _process(
            x: DataElement,
        ) -> ESM2DataElement:
            return ESM2DataElement(
                protein_id=x.protein_id,
                length=x.length,
                tokenized_seq=torch.tensor(self.tokenizer.encode(x.seq)),
                substructures=x.substructures,
                labels=x.labels,
            )
        super().__init__(source=source_node, map_fn=_process, num_workers=num_workers)


    def get_collate_fn(
        self,
        labels_mode: Literal["stack", "cat", "pad"],
    ) -> Callable:
        return partial(
            esm2_collate,
            pad_id=self.tokenizer.pad_token_id,
            labels_mode=labels_mode,
        )

def esm2_collate(
    entries: list[ESM2DataElement],
    pad_id: int,
    labels_mode: Literal["stack", "cat", "pad"],
) -> ESM2Batch:
    """
    Collate the entries into a batch.
    """
    protein_ids = [x.protein_id for x in entries]
    lengths = [x.length for x in entries]
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
        labels = [x.labels for x in entries]
        if labels_mode == "stack":
            labels = torch.stack(labels)
        elif labels_mode == "cat":
            labels = torch.cat(labels)
        elif labels_mode == "pad":
            labels = stack_variable_length_tensors(
                labels,
                constant_value=-1,
            )
        else:
            raise ValueError(f"unknown label mode: {labels_mode}")
    return ESM2Batch(
        protein_ids=protein_ids,
        lengths=lengths,
        tokenized_seq=padded_tensor,
        substructures=substructs,
        labels=labels,
    )