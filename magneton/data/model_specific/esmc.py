from pathlib import Path

from dataclasses import dataclass
from functools import partial
from typing import Callable, List

import torch

from esm.tokenization import get_esmc_model_tokenizers
from esm.utils.misc import stack_variable_length_tensors
from torchdata.nodes import BaseNode, ParallelMapper

from ..core import Batch, DataElement

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
        data_dir: str | Path = None,
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
        labels = torch.stack([x.labels for x in entries])
    return ESMCBatch(
        protein_ids=protein_ids,
        lengths=lengths,
        tokenized_seq=padded_tensor,
        substructures=substructs,
        labels=labels,
    )