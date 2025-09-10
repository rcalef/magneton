import bz2
import logging

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch

from pysam import FastaFile
from torchdata.nodes import BaseNode, ParallelMapper
from transformers import AutoTokenizer
from tqdm import tqdm

from esm.utils.misc import stack_variable_length_tensors

from magneton.data.core import Batch, DataElement
from magneton.utils import should_run_single_process

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass(kw_only=True)
class SaProtDataElement(DataElement):
    tokenized_sa_seq: torch.Tensor

@dataclass(kw_only=True)
class SaProtBatch(Batch):
    tokenized_sa_seq: torch.Tensor

    def to(self, device: str):
        super().to(device)
        self.tokenized_sa_seq = self.tokenized_sa_seq.to(device)
        return self

class SaProtTransformNode(ParallelMapper):
    def __init__(
        self,
        source_node: BaseNode,
        data_dir: str | Path,
        tokenizer_path: str | Path = "/home/rcalef/storage/om_storage/model_weights/SaProt_35M_AF2",
        num_workers: int = 2,
    ):
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        foldseek_tokens_path = data_dir / "foldseek_tokens.fa.bz2"
        if not foldseek_tokens_path.exists():
            raise ValueError(f"foldseek tokens not found: {foldseek_tokens_path}")

        # TODO (rcalef): add on-the-fly computation of foldseek tokens
        # if should_run_single_process() and not struct_tokens_path.exists():
        #     logger.info(
        #         f"ProSST structure tokens not found: {struct_tokens_path}\n"
        #         "Pre-computing. If data is large (>50k), using the standalone "
        #         "`compute_prosst_struct_toks.py` script may be better."
        #     )
        #     data_dir.mkdir(parents=True, exist_ok=True)
        #     precompute_struct_tokens(source_node, struct_tokens_path)
        # if dist.is_initialized():
        #     dist.barrier()

        logger.info(f"foldseek tokens file found at: {foldseek_tokens_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        self.foldseek_tokens = {}
        with bz2.open(foldseek_tokens_path, "rt") as fh:
            for i, line in enumerate(fh):
                if i % 2 == 0:
                    uniprot_id = line.strip()[1:]
                else:
                    self.foldseek_tokens[uniprot_id] = line.strip()

        logger.info(f"read foldseek structure tokens for {len(self.foldseek_tokens)} proteins")
        super().__init__(source=source_node, map_fn=self.process_example, num_workers=num_workers)

    def process_example(
        self,
        x: DataElement,
    ) -> SaProtDataElement:
        if x.protein_id not in self.foldseek_tokens:
            raise KeyError(f"{x.protein_id}: foldseek tokens not found: {x}")
        sa_seq = []
        foldseek_toks = self.foldseek_tokens[x.protein_id]
        assert len(foldseek_toks) == len(x.seq)
        for aa, fs in zip(x.seq, foldseek_toks):
            sa_seq.append(f"{aa}{fs}")
        sa_seq = "".join(sa_seq)
        return SaProtDataElement(
            protein_id=x.protein_id,
            length=x.length,
            tokenized_sa_seq=torch.tensor(self.tokenizer.encode(sa_seq)),
            substructures=x.substructures,
            labels=x.labels,
        )

    def get_collate_fn(
        self,
        stack_labels: bool = True,
    ) -> Callable:
        return partial(
            saprot_collate,
            pad_id=self.tokenizer.pad_token_id,
            stack_labels=stack_labels,
        )


def saprot_collate(
    entries: list[SaProtDataElement],
    pad_id: int,
    stack_labels: bool = True,
) -> SaProtBatch:
    """
    Collate ProSST data elements into a batch.
    """
    protein_ids = [x.protein_id for x in entries]
    lengths = [x.length for x in entries]

    padded_sa_seq_tensor = stack_variable_length_tensors(
        [x.tokenized_sa_seq for x in entries],
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
        if stack_labels:
            labels = torch.stack(labels)
        else:
            labels = torch.cat(labels)
    return SaProtBatch(
        protein_ids=protein_ids,
        lengths=lengths,
        tokenized_sa_seq=padded_sa_seq_tensor,
        substructures=substructs,
        labels=labels,
    )