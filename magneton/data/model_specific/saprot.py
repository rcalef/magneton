import bz2
import logging
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from Bio.PDB import PDBParser
from esm.utils.misc import stack_variable_length_tensors
from torchdata.nodes import BaseNode, ParallelMapper
from tqdm import tqdm
from transformers import AutoTokenizer

from magneton.data.core import Batch, DataElement
from magneton.utils import should_run_single_process

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class SaProtDataElement(DataElement):
    tokenized_sa_seq: torch.Tensor
    struct_seq: str


@dataclass(kw_only=True)
class SaProtBatch(Batch):
    tokenized_sa_seq: torch.Tensor
    struct_seqs: list[str]

    def to(self, device: str):
        super().to(device)
        self.tokenized_sa_seq = self.tokenized_sa_seq.to(device)
        return self


def extract_plddt(pdb_path: str) -> np.ndarray:
    """
    Extract plddt scores from pdb file.
    Args:
        pdb_path: Path to pdb file.

    Returns:
        plddts: plddt scores.
    """

    # Initialize parser
    parser = PDBParser()

    structure = parser.get_structure("protein", pdb_path)
    model = structure[0]
    chain = model["A"]

    # Extract plddt scores
    plddts = []
    for residue in chain:
        residue_plddts = []
        for atom in residue:
            plddt = atom.get_bfactor()
            residue_plddts.append(plddt)

        plddts.append(np.mean(residue_plddts))

    plddts = np.array(plddts)
    return plddts


def mask_foldseek(
    pdb_path: str,
    foldseek_seq: str,
    plddt_threshold: float = 70,
) -> str:
    plddts = extract_plddt(pdb_path)
    assert len(plddts) == len(foldseek_seq), (
        f"Length mismatch: {len(plddts)} != {len(foldseek_seq)}"
    )

    # Mask regions with plddt < threshold
    indices = np.where(plddts < plddt_threshold)[0]
    np_seq = np.array(list(foldseek_seq))
    np_seq[indices] = "#"
    return "".join(np_seq)


def run_one_job(
    line: str,
    pdb_dir: str,
) -> tuple[str, str]:
    name, _, foldseek_seq = line.strip().split("\t")[:3]
    protein_id = name.split()[0].replace(".pdb_A", "")

    full_path = f"{pdb_dir}/{protein_id}.pdb"
    masked_seq = mask_foldseek(full_path, foldseek_seq)

    return (protein_id, masked_seq)


def run_conversion(
    foldseek_path: str,
    output_path: str,
    pdb_dir: str,
    total: int | None = None,
    nprocs: int = 32,
):
    process_func = partial(run_one_job, pdb_dir=pdb_dir)
    with open(foldseek_path, "rt") as fh, Pool(nprocs) as p:
        results = list(tqdm(p.imap_unordered(process_func, fh), total=total))

    with bz2.open(output_path, "wt") as out_fh:
        for protein_id, masked_seq in results:
            out_fh.write(f">{protein_id}\n{masked_seq.lower()}\n")


def precompute_foldseek_tokens(
    data_source: BaseNode,
    output_path: Path,
    num_workers: int = 32,
):
    all_pdb_paths = []
    for data_elem in data_source:
        all_pdb_paths.append((data_elem.protein_id, Path(data_elem.structure_path)))
    num_pdbs = len(all_pdb_paths)
    logger.info(f"got {num_pdbs} pdb paths for FoldSeek tokens")

    foldseek_output_path = output_path.parent / f"{output_path.stem}.tsv"
    foldseek_output_dbtype_path = output_path.parent / f"{output_path.stem}.tsv.dbtype"

    # Temporary directory will be deleted automatically on exit
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create symlinks: protein_id.pdb -> actual structure file
        for protein_id, pdb_path in all_pdb_paths:
            link_path = tmpdir_path / f"{protein_id}.pdb"
            # There are occasionally duplicates in these datasets
            if not link_path.exists():
                link_path.symlink_to(pdb_path)

        # Run foldseek
        cmd = [
            "foldseek",
            "structureto3didescriptor",
            "--gpu",
            "1",
            "--threads",
            str(num_workers),
            "--chain-name-mode",
            "1",
            str(tmpdir_path),
            str(foldseek_output_path),
        ]
        logger.info("Running FoldSeek: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
        logger.info("FoldSeek completed. Processing for plddt masking.")
        run_conversion(
            foldseek_output_path,
            output_path,
            tmpdir_path,
            total=len(all_pdb_paths),
            nprocs=num_workers,
        )

    foldseek_output_path.unlink()
    # Also cleanup foldseek's bonus output
    foldseek_output_dbtype_path.unlink()

    logger.info("FoldSeek completed and temporary directory cleaned up.")


class SaProtTransformNode(ParallelMapper):
    def __init__(
        self,
        source_node: BaseNode,
        data_dir: str | Path,
        tokenizer_path: str
        | Path = "/home/rcalef/storage/om_storage/model_weights/SaProt_35M_AF2",
        num_workers: int = 2,
    ):
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        foldseek_tokens_path = data_dir / "foldseek_tokens.fa.bz2"
        # if not foldseek_tokens_path.exists():
        #     raise ValueError(f"foldseek tokens not found: {foldseek_tokens_path}")

        if should_run_single_process() and not foldseek_tokens_path.exists():
            logger.info(
                f"FoldSeek structure tokens not found: {foldseek_tokens_path}\n"
                "Pre-computing."
            )
            data_dir.mkdir(parents=True, exist_ok=True)
            precompute_foldseek_tokens(source_node, foldseek_tokens_path)
        if dist.is_initialized():
            dist.barrier()

        logger.info(f"foldseek tokens file found at: {foldseek_tokens_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )

        self.foldseek_tokens = {}
        with bz2.open(foldseek_tokens_path, "rt") as fh:
            for i, line in enumerate(fh):
                if i % 2 == 0:
                    uniprot_id = line.strip()[1:]
                else:
                    self.foldseek_tokens[uniprot_id] = line.strip()

        logger.info(
            f"read foldseek structure tokens for {len(self.foldseek_tokens)} proteins"
        )
        super().__init__(
            source=source_node, map_fn=self.process_example, num_workers=num_workers
        )

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
            seq=x.seq,
            substructures=x.substructures,
            labels=x.labels,
            tokenized_sa_seq=torch.tensor(self.tokenizer.encode(sa_seq)),
            struct_seq=foldseek_toks,
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
    seqs = [x.seq for x in entries]
    struct_seqs = [x.struct_seq for x in entries]

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
        substructures=substructs,
        labels=labels,
        seqs=seqs,
        tokenized_sa_seq=padded_sa_seq_tensor,
        struct_seqs=struct_seqs,
    )
