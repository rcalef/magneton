import json
import logging
from pathlib import Path
from typing import Any, Literal

import lmdb
import pandas as pd
import torch
from torch.utils.data import Dataset

from magneton.data.core import DataElement
from .utils import (
    download_afdb_files,
    parse_seqs_from_pdbs,
)
from .task_types import EVAL_TASK


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -------------------------
# LMDB reading utility
# -------------------------
def read_lmdb_folder(folder: Path) -> list[dict[str, Any]]:
    """
    Read an LMDB database stored in `folder` (folder should contain data.mdb / lock.mdb).
    Returns a list of JSON-decoded value dicts.

    Example:
        read_lmdb_folder(Path("Thermostability/normal/train"))
    """
    env = lmdb.open(str(folder), lock=False, readonly=True)
    with env.begin() as txn:
        # The DB uses 'length' key to store length (stored as bytes).
        length_b = txn.get(b"length")
        if length_b is None:
            raise RuntimeError(f"No 'length' key found in LMDB at {folder}")
        dataset_len = int(length_b.decode())

        got: list[dict[str, Any]] = []
        for i in range(dataset_len):
            val_b = txn.get(str(i).encode())
            if val_b is None:
                raise RuntimeError(f"missing key {i} in LMDB at {folder}; skipping")
            value_dict = json.loads(val_b.decode())
            got.append(value_dict)
    env.close()
    return got

def process_df(
    df: pd.DataFrame,
    fasta_cache_path: Path,
    struct_template: str,
    num_workers: int,
) -> pd.DataFrame:
    """Perform pre-processing for dataframe from LMDB with normalized names.

    In particular, this does the following:
    - adds in structure paths
    - downloads any missing PDB files from AlphaFoldDB
    - removes rows with no structures in AlphaFoldDB
    - removes rows where sequence in PDB file doesn't match the dataframe
    """
    # Attach structure paths
    df = df.assign(structure_path=lambda x: x.uniprot_id.map(lambda uid: Path(struct_template % uid)))

    # Download missing PDB files
    file_paths = df.structure_path.tolist()
    missing = [p for p in file_paths if not p.exists()]
    if len(missing) != 0:
        logger.info(f"downloading {len(missing)} / {len(file_paths)} missing AlphaFold PDBs")
        download_afdb_files(missing, num_workers=num_workers)

    # Drop entries without structures
    has_struct = df.structure_path.apply(lambda p: p.exists())
    if not has_struct.all():
        n_drop = (~has_struct).sum()
        logger.warning(f"Dropping {n_drop} entries with missing AFDB structures")
        df = df.loc[has_struct].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("All entries dropped because AFDB structures are missing")

    # Parse sequences from PDBs and verify they match supplied sequences
    logger.info("Extracting sequences from PDBs to verify dataset sequences.")
    seq_from_pdb: dict[str, str] = parse_seqs_from_pdbs(
        fasta_path=fasta_cache_path,
        file_paths=df.structure_path.tolist(),
        uniprot_ids=df.uniprot_id.tolist(),
        num_workers=num_workers,
    )

    df = df.assign(seq_pdb=lambda x: x.uniprot_id.map(seq_from_pdb))
    match = df.apply(lambda r: r.seq == r.seq_pdb, axis=1)
    if not match.all():
        n_mismatch = (~match).sum()
        logger.warning(f"Dropping {n_mismatch} entries with sequence mismatches vs PDB")
        df = df.loc[match].reset_index(drop=True)
    return df


class SaProtProcessedDataset(Dataset):
    """
    Torch Dataset that returns DataElement instances from eval datasets originally
    parsed by the SaProt authors.

    Expected df columns: ['uniprot_id', 'seq', 'label', 'structure_path']
    label should be a numeric value (float).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        task_type: EVAL_TASK,
        num_classes: int | None = None
    ):
        if task_type in [EVAL_TASK.BINARY, EVAL_TASK.MULTICLASS]:
            assert num_classes is not None
        self.df = df.reset_index(drop=True)
        self.task_type = task_type
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> DataElement:
        row = self.df.iloc[idx]
        # labels stored as 1-element float tensor
        if self.task_type == EVAL_TASK.REGRESSION:
            label_tensor = torch.tensor([float(row.label)], dtype=torch.float32)
        elif self.task_type == EVAL_TASK.BINARY:
            label_tensor = torch.tensor(float(row.label), dtype=torch.int32)
        elif self.task_type == EVAL_TASK.MULTICLASS:
            label_tensor = torch.tensor(float(row.label), dtype=torch.int32)
        elif self.task_type == EVAL_TASK.MULTILABEL:
            # (N X num_classes) -> (num_classes)
            # where N is the number of labels this item is anntoated with
            label_tensor = torch.nn.functional.one_hot(
                torch.tensor(row.label),
                num_classes=self.num_classes,
            ).sum(dim=0)

        return DataElement(
            protein_id=row.uniprot_id,
            length=len(row.seq),
            labels=label_tensor,
            seq=row.seq,
            structure_path=str(row.structure_path),
        )


# -------------------------
# Thermostability module
# -------------------------
class ThermostabilityModule:
    """
    Load Thermostability LMDB datasets from SaProt. These are structured in the pattern:
      <dataset_root>/normal/<split>/
    where each split folder contains LMDB files (data.mdb, lock.mdb).

    Usage:
      mod = ThermostabilityModule(data_dir="Thermostability", struct_template="/path/to/afdb/%s.pdb")
      train_ds = mod.get_dataset("train")
    """

    def __init__(
        self,
        data_dir: str | Path,
        struct_template: str,
        num_workers: int = 16,
    ):
        self.data_dir = Path(data_dir)
        self.struct_template = struct_template
        self.num_workers = num_workers

    def _split_folder(self, split: Literal["train", "val", "test"]) -> Path:
        if split == "val":
            split = "valid"
        return self.data_dir / "normal" / split

    def num_classes(self) -> int:
        return 1

    def get_dataset(self, split: Literal["train", "val", "test"]) -> Dataset:
        df = self._prepare_split_df(split)
        return SaProtProcessedDataset(df=df, task_type=EVAL_TASK.REGRESSION)

    def _prepare_split_df(self, split: Literal["train", "val", "test"]) -> pd.DataFrame:
        folder = self._split_folder(split)
        if not folder.exists():
            raise FileNotFoundError(f"LMDB split folder not found: {folder}")

        records = read_lmdb_folder(folder)
        logger.info(f"Read {len(records)} records from LMDB at {folder}")
        assert len(records) != 0

        # Normalize and convert into DataFrame
        df = (
            pd.DataFrame(records)
            .rename(columns={
                "name": "uniprot_id",
                "fitness": "label",
            })
        )
        df = process_df(
            df=df,
            fasta_cache_path=self.data_dir / f"thermostability.seq_from_pdb.{split}.fa",
            struct_template=self.struct_template,
            num_workers=self.num_workers,
        )
        # The processed dataset contains some duplicated items which have slightly different labels
        # presumably from different runs in the original dataset
        df = df.drop_duplicates("uniprot_id")

        # Final columns expected by LMDBRegressionDataset
        df = df[["uniprot_id", "seq", "label", "structure_path"]]
        logger.info(f"Prepared {len(df)} entries for split {split}")
        return df


# -------------------------
# Thermostability module
# -------------------------
class DeepLocModule:
    """
    Load DeepLoc LMDB datasets from SaProt, These are structured in the pattern:
      <dataset_root>/<num_classes>/normal/<split>/
    where each split folder contains LMDB files (data.mdb, lock.mdb) and
    num_classes can be `cls10` or `cls2` for multiclass or binary classification
    respectively.

    Usage:
      mod = DeepLocModule(data_dir="/path/to/magneton-data/DeepLoc", struct_template="/path/to/afdb/%s.pdb")
      train_ds = mod.get_dataset("train")
    """

    def __init__(
        self,
        data_dir: str | Path,
        struct_template: str,
        num_labels: Literal[2, 10],
        num_workers: int = 16,
    ):
        self.data_dir = Path(data_dir)
        self.struct_template = struct_template
        self.num_workers = num_workers
        assert num_labels in [2, 10], f"bad number of labels {self.num_labels}"
        self.num_labels = num_labels

    def _split_folder(self, split: Literal["train", "val", "test"]) -> Path:
        if split == "val":
            split = "valid"
        return self.data_dir / f"cls{self.num_labels}" / "normal" / split

    def num_classes(self) -> int:
        if self.num_labels == 2:
            # Not ideal, but this is being used to set the number of outputs
            # for the head classifier, so we really just want one output
            # if it's binary classification
            return 1
        return self.num_labels

    def get_dataset(self, split: Literal["train", "val", "test"]) -> Dataset:
        df = self._prepare_split_df(split)
        if self.num_labels == 2:
            task_type = EVAL_TASK.BINARY
        elif self.num_labels == 10:
            task_type = EVAL_TASK.MULTICLASS
        else:
            # This should never happen
            raise ValueError(f"bad number of labels: {self.num_labels}")

        return SaProtProcessedDataset(df=df, task_type=task_type, num_classes=self.num_labels)

    def _prepare_split_df(self, split: Literal["train", "val", "test"]) -> pd.DataFrame:
        folder = self._split_folder(split)
        if not folder.exists():
            raise FileNotFoundError(f"LMDB split folder not found: {folder}")

        records = read_lmdb_folder(folder)
        logger.info(f"Read {len(records)} records from LMDB at {folder}")
        assert len(records) != 0

        # Normalize and convert into DataFrame
        df = (
            pd.DataFrame(records)
            .rename(columns={
                "name": "uniprot_id",
            })
        )

        df = process_df(
            df=df,
            fasta_cache_path=self.data_dir / f"deeploc.cls{self.num_labels}.seq_from_pdb.{split}.fa",
            struct_template=self.struct_template,
            num_workers=self.num_workers,
        )

        # Final columns expected by LMDBRegressionDataset
        df = df[["uniprot_id", "seq", "label", "structure_path"]]
        logger.info(f"Prepared {len(df)} entries for split {split}")
        return df
