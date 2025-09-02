import bz2
import sys
import logging

from pathlib import Path

import fire
import hydra
import torch

from hydra.utils import instantiate
from tqdm import tqdm

from magneton.config import PipelineConfig, DataConfig
from magneton.data.core import CoreDataset
from magneton.data import SupervisedDownstreamTaskDataModule
from magneton.types import DataType


PROSST_REPO_PATH = (
    Path(__file__).parent.parent /
    "magneton" /
    "external" /
    "ProSST"
)
print(PROSST_REPO_PATH)
sys.path.append(str(PROSST_REPO_PATH))
from prosst.structure.get_sst_seq import SSTPredictor, init_shared_pool

logger = logging.Logger(__file__)
logger.setLevel(logging.INFO)


def compute_prosst_toks(
    output_path: str,
    data_dir: str,
    prefix: str,
    fasta_path: str,
    struct_template: str,
    eval_task: str | None = None,
    resume_path: str| None = None,
    pdbs_batch_size: int = 128,
    max_len: int | None = None,
    shard_num: int | None = None,
    num_shards: int | None = None
):
    """Compute ProSST structure tokens for all PDB files in a directory.

    pdb_dir_path: Directory containing .pdb files
    output_path: Path to write TSV
    filter_path: Path to
    """
    assert (shard_num is None) == (num_shards is None)

    data_config = DataConfig(
        data_dir=data_dir,
        prefix=prefix,
        fasta_path=fasta_path,
        struct_template=struct_template,
    )

    if eval_task is not None:
        module = SupervisedDownstreamTaskDataModule(
            data_config=data_config,
            task=eval_task,
            data_dir=data_dir,
            model_type="prosst",
        )
        dataset = module.module.get_dataset("train")

    else:
        dataset = CoreDataset(
            data_config=data_config,
            want_datatypes=[DataType.STRUCT, DataType.SEQ],
        )

    print(f"collecting PDB paths from dataset: {len(dataset)}")
    all_pdb_paths = []
    for prot in tqdm(dataset):
        if max_len is not None and len(prot.seq) >= max_len:
            continue
        all_pdb_paths.append((prot.protein_id, prot.structure_path))

    num_pdbs = len(all_pdb_paths)
    print(f"got {num_pdbs} pdb paths")
    print(all_pdb_paths[:5])

    if num_shards is not None:
        prots_per_shard = num_pdbs // num_shards
        my_start = prots_per_shard * shard_num
        if shard_num == (num_shards-1):
            my_end = num_pdbs
        else:
            my_end = my_start + prots_per_shard
        all_pdb_paths = all_pdb_paths[my_start:my_end]
        num_pdbs = len(all_pdb_paths)
        print(f"shard {shard_num} / {num_shards}, running [{my_start}:{my_end}]")

    if resume_path is not None:
        keep = []
        with bz2.open(resume_path, "rt") as fh:
            present = {l.split("\t")[0] for l in fh}
        for (uniprot_id, path) in all_pdb_paths:
            if uniprot_id in present:
                continue
            keep.append((uniprot_id, path))
        print(f"resuming with {len(keep)} / {len(all_pdb_paths)} remaining")
        all_pdb_paths = keep
        num_pdbs = len(all_pdb_paths)

    predictor = SSTPredictor(
        structure_vocab_size=2048,
        num_processes=4,
        num_threads=32,
    )
    num_batches = (num_pdbs + pdbs_batch_size - 1) // pdbs_batch_size
    start = 0
    end = min(num_pdbs, pdbs_batch_size)

    with bz2.open(output_path, "wt") as fh, torch.inference_mode():
        for _ in tqdm(range(num_batches)):
            batch = all_pdb_paths[start:end]
            uniprot_ids, paths = zip(*batch)
            results = predictor.predict_from_pdb(paths)

            for uniprot_id, result in zip(uniprot_ids, results):
                struct_tok_strs = " ".join(map(str, result["2048_sst_seq"]))
                print(f"{uniprot_id}\t{struct_tok_strs}", file=fh)

            start = end
            end = min(start + pdbs_batch_size, num_pdbs)



if __name__ == "__main__":
    init_shared_pool(32)
    fire.Fire(compute_prosst_toks)
