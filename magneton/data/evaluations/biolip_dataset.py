import gzip
import json
import logging
import tarfile
from functools import partial
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.PDB import PDBParser
from torch.utils.data import Dataset

from magneton.data.core import DataElement
from magneton.data.evaluations.utils import parse_seqs_from_pdbs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BioLIP2Dataset(Dataset):
    """
    Torch Dataset reading a single BioLIP2 jsonl split.
    Each item -> DataElement with per-residue binary labels.
    """

    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> DataElement:
        row = self.dataset.iloc[idx]

        return DataElement(
            protein_id=row.protein_id,
            length=len(row.seq),
            labels=row.labels,
            seq=row.seq,
            structure_path=str(row.structure_path),
        )


class BioLIP2Module:
    """
    Data module for the BioLIP2 functional site prediction benchmark.

    Directory is expected to contain files such as:
        BioLIP2FunctionDataset_binding_label_train.jsonl
        BioLIP2FunctionDataset_binding_label_validation.jsonl
        BioLIP2FunctionDataset_binding_label_fold_test.jsonl
        BioLIP2FunctionDataset_binding_label_superfamily_test.jsonl
        BioLIP2FunctionDataset_catalytic_label_train.jsonl
        ...

    Parameters
    ----------
    data_dir : str or Path
        Directory containing the jsonl files.
    task : {"binding","catalytic"}
        Whether to load the binding-site or catalytic-site labels.
    """

    def __init__(
        self,
        data_dir: str | Path,
        task: Literal["binding", "catalytic"],
        unk_amino_acid_char: str = "X",
        num_workers: int = 16,
    ):
        self.data_dir = Path(data_dir)
        self.num_workers = num_workers
        self.task = task
        self.unk_amino_acid_char = unk_amino_acid_char

        if task not in {"binding", "catalytic"}:
            raise ValueError(f"task must be 'binding' or 'catalytic', got: {task}")

        # The following interact poorly with FoldSeek and other tokenizers
        # due to having residues missing alpha-carbon atoms
        self.exclude = [
            "5wdh_A",
            "3zqe_A",
            "1jni_A",
            "5ivn_A",
            "2bp3_A",
            "2a06_G",
            "4ku4_A",
            "1plj_A",
            "1brd_A",
            "2vn2_A",
            "2zba_D",
        ]

        # Possibly unpack tarfile
        expected_structure_path = self.data_dir / "biolip2" / "binding"
        if not expected_structure_path.exists():
            logger.info("BioLIP2 PDB files not found, attempting to unpack tar file")
            tar_path = self.data_dir / "biolip2" / "pdb_files.tar.gz"
            with tarfile.open(tar_path, "r") as tar:
                tar.extractall(path=self.data_dir / "biolip2")
            logger.info("BioLIP2 PDB file tar unpacked")


    def _jsonl_name(self, split: str) -> Path:
        """
        Map split to jsonl file name.
        Valid splits include: train, validation, test, fold_test, superfamily_test.
        """
        # standardise known split names
        valid_splits = {
            "train": "train",
            "val": "validation",
            "test": "fold_test",
            "fold_test": "fold_test",
            "superfamily_test": "superfamily_test",
        }
        if split not in valid_splits:
            raise ValueError(f"split must be one of {list(valid_splits)}")

        return (
            self.data_dir
            / f"BioLIP2FunctionDataset_{self.task}_label_{valid_splits[split]}.jsonl.gz"
        )

    def num_classes(self) -> int:
        return 1

    def get_dataset(
        self,
        split: Literal["train", "validation", "test", "fold_test", "superfamily_test"],
    ) -> Dataset:
        json_path = self._jsonl_name(split)
        if not json_path.exists():
            raise FileNotFoundError(f"Expected jsonl file not found: {json_path}")

        label_key = f"{self.task}_label"

        all_protein_ids = []
        all_paths = []
        all_labels = []
        all_residue_idxs = []
        with gzip.open(json_path, "rt") as fh:
            for line in fh:
                record = json.loads(line)
                path = self.data_dir / record["pdb_path"]
                protein_id = path.name.split(".")[0]
                labels = record[label_key]
                residue_idxs = record["residue_index"]
                assert path.exists(), f"path not found: {path}"
                assert len(labels) == len(residue_idxs)

                all_protein_ids.append(protein_id)
                all_paths.append(path)
                all_labels.append(torch.tensor(labels))
                all_residue_idxs.append(residue_idxs)

        logger.info(f"Loaded {len(all_protein_ids)} records from {json_path}")

        df = pd.DataFrame(
            {
                "protein_id": all_protein_ids,
                "structure_path": all_paths,
                "labels": all_labels,
            }
        )

        fasta_path = (
            self.data_dir
            / f"{self.task}.{split}.seqs_from_pdbs_unk{self.unk_amino_acid_char}.fa"
        )
        sequence_dict = parse_seqs_from_pdbs(
            fasta_path=fasta_path,
            jobs=df.structure_path.to_list(),
            protein_ids=df.protein_id.to_list(),
            num_workers=self.num_workers,
            parse_func=partial(
                parse_seq_direct, unk_amino_acid_char=self.unk_amino_acid_char
            ),
        )
        df = df.assign(seq=lambda x: x.protein_id.map(sequence_dict))
        num_labels = df.labels.map(lambda x: x.size(dim=0))
        mismatch_len = df.seq.str.len() != num_labels
        if mismatch_len.sum() != 0:
            logger.info(
                f"Removing {mismatch_len.sum()} / {len(df)} records with mismatched sequence and labels"
            )
            df = df.loc[~mismatch_len]

        duplicated = df[["protein_id", "structure_path"]].duplicated()
        logger.info(f"Removing {duplicated.sum()} / {len(df)} duplicated records")
        df = df.loc[~duplicated]

        df = df.loc[lambda x: ~x.protein_id.isin(self.exclude)]
        logger.info(f"Final dataset of {len(df)} records after removing bad samples")

        return BioLIP2Dataset(df)


def parse_seq_direct(
    path: Path,
    unk_amino_acid_char: str,
) -> str:
    """Directly parse a sequence from a PDB file.

    Instead of using BioPython's PPBuilder, just directly read the amino acid
    identities from the file. Some of the PDB files here seem to have missing
    atoms that result in omitted residues otherwise.
    """
    aa_map = {k.upper(): v for k, v in protein_letters_3to1.items()}
    aa_map["UNK"] = unk_amino_acid_char
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", path)

    # AFDB structures all consist of one chain, so just take the first one.
    # This seems to be what folks must do for using these classic splits based
    # on PDB IDs (both ProSST and SaProt use AFDB structures instead of the
    # available PDB structures).
    chains = list(list(structure)[0])
    if len(chains) != 1:
        raise ValueError(f"expected single chain, got {len(chains)}: {path}")
    try:
        return "".join([aa_map[res.resname] for res in chains[0].get_residues()])
    except KeyError as e:
        print(path)
        raise e
