import bz2
import logging
import sys

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist

from torchdata.nodes import BaseNode, ParallelMapper
from transformers import AutoTokenizer
from tqdm import tqdm

from esm.utils.misc import stack_variable_length_tensors

from magneton.data.core import Batch, DataElement
from magneton.utils import should_run_single_process

PROSST_REPO_PATH = (
    Path(__file__).parent.parent.parent /
    "external" /
    "ProSST"
)
sys.path.append(str(PROSST_REPO_PATH))
from prosst.structure.get_sst_seq import SSTPredictor, init_shared_pool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def precompute_struct_tokens(
    data_source: BaseNode,
    output_path: Path,
    batch_size: int = 128,
    max_len: int = 2048,
    num_workers: int = 32,
):
    init_shared_pool(num_workers)
    all_pdb_paths = []
    for data_elem in data_source:
        all_pdb_paths.append((data_elem.protein_id, data_elem.structure_path))
    num_pdbs = len(all_pdb_paths)
    logger.info(f"got {num_pdbs} pdb paths")

    predictor = SSTPredictor(
        structure_vocab_size=2048,
        num_processes=1,
    )
    num_batches = (num_pdbs + batch_size - 1) // batch_size
    start = 0
    end = min(num_pdbs, batch_size)

    with bz2.open(output_path, "wt") as fh, torch.inference_mode():
        for _ in tqdm(range(num_batches)):
            batch = all_pdb_paths[start:end]
            uniprot_ids, paths = zip(*batch)
            results = predictor.predict_from_pdb(paths)

            for uniprot_id, result in zip(uniprot_ids, results):
                struct_tok_strs = " ".join(map(str, result["2048_sst_seq"]))
                print(f"{uniprot_id}\t{struct_tok_strs}", file=fh)

            start = end
            end = min(start + batch_size, num_pdbs)


@dataclass(kw_only=True)
class ProSSTDataElement(DataElement):
    tokenized_seq: torch.Tensor
    tokenized_struct: torch.Tensor

@dataclass(kw_only=True)
class ProSSTBatch(Batch):
    tokenized_seq: torch.Tensor
    tokenized_struct: torch.Tensor

    def to(self, device: str):
        super().to(device)
        self.tokenized_seq = self.tokenized_seq.to(device)
        self.tokenized_struct = self.tokenized_struct.to(device)
        return self

class ProSSTTransformNode(ParallelMapper):
    def __init__(
        self,
        source_node: BaseNode,
        data_dir: str | Path,
        tokenizer_path: str | Path = "/home/rcalef/storage/om_storage/model_weights/ProSST-2048",
        num_workers: int = 2,
    ):
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        struct_tokens_path = data_dir / "prosst_toks.tsv.bz2"

        if should_run_single_process() and not struct_tokens_path.exists():
            logger.info(
                f"ProSST structure tokens not found: {struct_tokens_path}\n"
                "Pre-computing. If data is large (>50k), using the standalone "
                "`compute_prosst_struct_toks.py` script may be better."
            )
            data_dir.mkdir(parents=True, exist_ok=True)
            precompute_struct_tokens(source_node, struct_tokens_path)
        if dist.is_initialized():
            dist.barrier()

        logger.info(f"ProSST tokens file found at: {struct_tokens_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        self.struct_tokens = {}
        with bz2.open(struct_tokens_path, "rt") as fh:
            for i, line in enumerate(fh):
                parts = line.split("\t")
                assert len(parts) == 2, f"bad line at {i}: {line}"
                uniprot_id, tokens = parts
                # +3 here is taken from ProSST author's notebook for variant effect prediction,
                # see cell 6 here:
                #  https://github.com/ai4protein/ProSST/blob/main/zero_shot/score_mutant.ipynb
                raw_tokens = list(map(lambda x: int(x)+3, tokens.split()))
                # Similar to above, from same notebook,  prepending 1 and appending 2 tokens.
                # These correspond to the CLS and EOS tokens from their tokenizer, which
                # is in fact different from ESM's tokenizer.
                modified_tokens = [self.tokenizer.cls_token_id] + raw_tokens + [self.tokenizer.eos_token_id]
                self.struct_tokens[uniprot_id] = torch.tensor(modified_tokens)

        logger.info(f"read ProSST structure tokens for {len(self.struct_tokens)} proteins")

        super().__init__(source=source_node, map_fn=self.process_example, num_workers=num_workers)

    def process_example(
        self,
        x: DataElement,
    ) -> ProSSTDataElement:
        if x.protein_id not in self.struct_tokens:
            raise KeyError(f"{x.protein_id}: ProSST tokens not found: {x}")
        return ProSSTDataElement(
            protein_id=x.protein_id,
            length=x.length,
            tokenized_seq=torch.tensor(self.tokenizer.encode(x.seq)),
            tokenized_struct=self.struct_tokens[x.protein_id],
            substructures=x.substructures,
            labels=x.labels,
        )

    def get_collate_fn(
        self,
        stack_labels: bool = True,
    ) -> Callable:
        return partial(
            prosst_collate,
            pad_id=self.tokenizer.pad_token_id,
            stack_labels=stack_labels,
        )


def prosst_collate(
    entries: list[ProSSTDataElement],
    pad_id: int,
    stack_labels: bool = True,
) -> ProSSTBatch:
    """
    Collate ProSST data elements into a batch.
    """
    protein_ids = [x.protein_id for x in entries]
    lengths = [x.length for x in entries]

    padded_seq_tensor = stack_variable_length_tensors(
        [x.tokenized_seq for x in entries],
        constant_value=pad_id,
    )
    padded_struct_tensor = stack_variable_length_tensors(
        [x.tokenized_struct for x in entries],
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
    return ProSSTBatch(
        protein_ids=protein_ids,
        lengths=lengths,
        tokenized_seq=padded_seq_tensor,
        tokenized_struct=padded_struct_tensor,
        substructures=substructs,
        labels=labels,
    )