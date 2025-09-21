import json
import logging
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Literal

import lmdb
import numpy as np
import pandas as pd
import requests
import torch

from Bio import SeqIO
from scipy.spatial.distance import pdist, squareform
from torch.utils.data import Dataset
from tqdm import tqdm

from magneton.data.core import DataElement

from .task_types import EVAL_TASK
from .utils import (
    download_afdb_files,
    parse_seqs_from_pdbs,
)

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
    df = df.assign(
        structure_path=lambda x: x.uniprot_id.map(
            lambda uid: Path(struct_template % uid)
        )
    )

    # Download missing PDB files
    file_paths = df.structure_path.tolist()
    missing = [p for p in file_paths if not p.exists()]
    if len(missing) != 0:
        logger.info(
            f"downloading {len(missing)} / {len(file_paths)} missing AlphaFold PDBs"
        )
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
        self, df: pd.DataFrame, task_type: EVAL_TASK, num_classes: int | None = None
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
        df = pd.DataFrame(records).rename(
            columns={
                "name": "uniprot_id",
                "fitness": "label",
            }
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

        return SaProtProcessedDataset(
            df=df, task_type=task_type, num_classes=self.num_labels
        )

    def _prepare_split_df(self, split: Literal["train", "val", "test"]) -> pd.DataFrame:
        folder = self._split_folder(split)
        if not folder.exists():
            raise FileNotFoundError(f"LMDB split folder not found: {folder}")

        records = read_lmdb_folder(folder)
        logger.info(f"Read {len(records)} records from LMDB at {folder}")
        assert len(records) != 0

        # Normalize and convert into DataFrame
        df = pd.DataFrame(records).rename(
            columns={
                "name": "uniprot_id",
            }
        )

        df = process_df(
            df=df,
            fasta_cache_path=self.data_dir
            / f"deeploc.cls{self.num_labels}.seq_from_pdb.{split}.fa",
            struct_template=self.struct_template,
            num_workers=self.num_workers,
        )

        # Final columns expected by LMDBRegressionDataset
        df = df[["uniprot_id", "seq", "label", "structure_path"]]
        logger.info(f"Prepared {len(df)} entries for split {split}")
        return df


class ContactPredictionDataset(Dataset):
    """
    Torch Dataset that returns DataElement instances from eval datasets originally
    parsed by the SaProt authors.

    Expected df columns: ['uniprot_id', 'seq', 'label', 'structure_path']
    label should be a numeric value (float).
    """

    def __init__(
        self,
        df: pd.DataFrame,
    ):
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> DataElement:
        row = self.df.iloc[idx]

        return DataElement(
            protein_id=row.protein_id,
            length=len(row.seq),
            labels=row.labels.float(),
            seq=row.seq,
            structure_path=str(row.structure_path),
        )

class ContactPredictionModule:
    """
    Load Contact prediction LMDB datasets from SaProt, These are structured in the pattern:
      <dataset_root>/<num_classes>/normal/<split>/
    where each split folder contains LMDB files (data.mdb, lock.mdb)
    Corresponding PDB files will be automatically downloaded

    Usage:
      mod = ContactPredictionModule(data_dir="/path/to/magneton-data/DeepLoc",)
      train_ds = mod.get_dataset("train")
    """

    def __init__(
        self,
        data_dir: str | Path,
        max_len: int = 1024,
        num_workers: int = 16,
    ):
        self.data_dir = Path(data_dir)
        self.pdb_dir = self.data_dir / "cached_pdbs"
        self.pdb_dir.mkdir(exist_ok=True)

        self.num_workers = num_workers
        self.max_len = max_len

    def _split_folder(self, split: Literal["train", "val", "test"]) -> Path:
        if split == "val":
            split = "valid"
        return self.data_dir / "normal" / split

    def num_classes(self) -> int:
        # Not ideal, but this is being used to set the number of outputs
        # for the head classifier, so we really just want one output
        # if it's binary classification
        return 1

    def get_dataset(self, split: Literal["train", "val", "test"]) -> Dataset:
        df = self._prepare_data(split)
        return ContactPredictionDataset(df=df)

    def _prepare_data(self, split: Literal["train", "val", "test"]) -> pd.DataFrame:
        folder = self._split_folder(split)
        if not folder.exists():
            raise FileNotFoundError(f"LMDB split folder not found: {folder}")

        records = read_lmdb_folder(folder)
        logger.info(f"Read {len(records)} records from LMDB at {folder}")
        assert len(records) != 0

        all_usable_records = []
        for record in records:
            # For now, it seems like fetching the domain-level PDB files
            # from SCOPe is prohibitively slow, so just skip the IDs that
            # look like {pdb_id}_{scope_domain}, and just use the ones that
            # look like {pdb_id}_{entity_num}_{chain_id}
            name = record["name"]
            if "#" in name:
                name = name.split("#")[1]
            parts = name.split("_")
            if split in["train", "val"] and len(parts) != 3:
                continue
            record["name"] = name

            all_usable_records.append(record)

        valid_seqs = []
        for record in all_usable_records:
            seq = []
            for aa, valid in zip(record["seq"], record["valid_mask"]):
                if valid:
                    seq.append(aa)
            valid_seqs.append("".join(seq))

        logger.info(f"Making contact labels for {len(all_usable_records)} records")
        labels = list(tqdm(map(record_to_labels, all_usable_records)))

        df = (
            pd.DataFrame({
                "protein_id": [x["name"] for x in all_usable_records],
                "seq": [x["seq"] for x in all_usable_records],
                # valid_seq is the masked sequence, which we use to do a sanity
                # check against the sequence in the PDB files. The test set PDB files
                # don't contain entries for masked positions.
                "valid_seq": valid_seqs,
                "labels": labels,
            })
            .assign(
                pdb_id=lambda x: x.protein_id.str.split("_").str[0],
            )
        )
        if split == "test":
            want_dir = self.data_dir / "test_set"
        else:
            want_dir = self.data_dir / "cached_pdbs"
            df = df.assign(
                model_num=lambda x: x.protein_id.str.split("_").str[1].astype(int),
                chain=lambda x: x.protein_id.str.split("_").str[2],
            )

        df = df.assign(
            structure_path=lambda x: x.pdb_id.apply(lambda id: want_dir / f"{id}.pdb"),
        )

        df = df.loc[lambda x: x.seq.str.len() <= self.max_len]
        logger.info(f"Retaining {len(df)} records after filtering for length <= {self.max_len}")

        # Try to download any missing files
        missing_files = [path for path in df.structure_path if not path.exists()]
        if len(missing_files) != 0:
            logger.info(f"Downloading missing {len(missing_files)} PDB files")
            download_pdb_files(missing_files, num_workers=self.num_workers)

        # Drop entries without structures
        has_struct = df.structure_path.apply(lambda p: p.exists())
        if not has_struct.all():
            n_drop = (~has_struct).sum()
            logger.warning(f"Dropping {n_drop} entries with missing PDB structures")
            df = df.loc[has_struct].reset_index(drop=True)
        if df.empty:
            raise RuntimeError("All entries dropped because PDB structures are missing")

        # Parse sequences from PDBs and verify they match supplied sequences
        logger.info("Extracting sequences from PDBs to verify dataset sequences.")
        fasta_cache_path = self.data_dir / f"contact.seq_from_pdb.{split}.fa"

        if split in ["train", "val"]:
            jobs = list(df[["structure_path", "chain"]].itertuples(index=False, name=None))
            want_dir = self.data_dir / "test_set"
        else:
            jobs = df.structure_path.tolist()

        seq_from_pdb: dict[str, str] = parse_seqs_from_pdbs(
            fasta_path=fasta_cache_path,
            jobs=jobs,
            protein_ids=df.protein_id.tolist(),
            num_workers=self.num_workers,
            parse_func=parse_chain_seq_from_pdb,
        )

        df = df.assign(seq_pdb=lambda x: x.protein_id.map(seq_from_pdb))
        if split in ["train", "val"]:
            match = df.apply(lambda r: r.seq == r.seq_pdb, axis=1)
        else:
            match = df.apply(lambda r: r.valid_seq == r.seq_pdb, axis=1)
        if not match.all():
            n_mismatch = (~match).sum()
            logger.warning(f"Dropping {n_mismatch} entries with sequence mismatches vs PDB")
            df = df.loc[match].reset_index(drop=True)

        return df

def parse_chain_seq_from_pdb(
    path: Path,
    chain: str = "A",
) -> str:
    """Get a sequence of a specific chain from a PDB file."""
    chains = {record.id: record.seq for record in SeqIO.parse(path, 'pdb-seqres')}
    name = path.name.split(".")[0]
    return str(chains.get(f"{name}:{chain}", ""))

def record_to_labels(
    entry: dict[str, Any],
    max_len: int = 2048,
) -> torch.Tensor:
    # Below is copied from SaProt:
    #  https://github.com/westlake-repl/SaProt/blob/main/dataset/saprot/saprot_contact_dataset.py#L56
    valid_mask = np.array(entry['valid_mask'])[:max_len]
    coords = np.array(entry['tertiary'])[:max_len]
    contact_map = np.less(squareform(pdist(coords)), 8.0).astype(np.int64)

    y_inds, x_inds = np.indices(contact_map.shape)
    invalid_mask = ~(valid_mask[:, None] & valid_mask[None, :])
    invalid_mask |= np.abs(y_inds - x_inds) < 6
    contact_map[invalid_mask] = -1
    return torch.tensor(contact_map)


def download_one_pdb_file(
    path: Path,
) -> bool:
    base_url = "https://files.rcsb.org/download/"
    url = base_url + path.name

    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            with open(path, "wb") as f:
                f.write(r.content)
        else:
            logger.debug(f"failed to fetch PDB file ({path}): status code {r.status_code}")
            return False
    except Exception as e:
        logger.debug(f"failed to fetch PDB file ({path}): {e}")
        return False
    return True


def download_pdb_files(
    file_paths: list[Path],
    num_workers: int,
):
    with Pool(num_workers) as p:
        download_results = list(
            tqdm(
                p.imap_unordered(download_one_pdb_file, file_paths),
                total=len(file_paths),
            )
        )
    success = sum(download_results)
    logger.info(f"succesfully downloaded {success}  / {len(file_paths)} files")
