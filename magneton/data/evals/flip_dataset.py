# flip_dataset.py
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Literal, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from magneton.data.core import DataElement
from .utils import (
    download_afdb_files,
    parse_seqs_from_pdbs,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Label channel configuration
# -----------------------------
# Legend (provided):
# 000 = No binding
# 001 = Metal
# 010 = Nucleic
# 011 = Nucleic + Metal
# 100 = Small
# 101 = Small + Metal
# 110 = Small + Nucleic
# 111 = Small + Nucleic + Metal
#
# Bit order assumptions (LSB -> MSB): Metal (b0), Nucleic (b1), Small (b2)
# channel order we expose by default:
DEFAULT_LABEL_TYPES: tuple[str, ...] = ("metal", "nucleic", "small")
BIT_INDEX_FOR_LABEL: Dict[str, int] = {"metal": 0, "nucleic": 1, "small": 2}

# -----------------------------
# FASTA parsing
# -----------------------------
@dataclass(frozen=True)
class LabelRecord:
    uniprot_id: str
    set_name: str          # e.g., "train" or "test"
    is_validation: bool    # from 'VALIDATION=' field
    labels_compact: str    # string of digits, possibly spanning multiple lines


FASTA_HEADER_RE = re.compile(
    r"^>(?P<uid>\S+)(?:\s+SET=(?P<set>\w+)\s+VALIDATION=(?P<val>\w+))?$",
    flags=re.IGNORECASE,
)


def _read_fasta(path: Path) -> Iterator[tuple[str, str]]:
    """Generic FASTA reader yielding (header, sequence_string) with newlines removed."""
    header = None
    chunks: list[str] = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line
                chunks = []
            else:
                chunks.append(line)
        if header is not None:
            yield header, "".join(chunks)


def read_sequence_fasta(path: Path) -> Dict[str, str]:
    """Parse sequences.fasta mapping UniProt -> amino-acid sequence."""
    seqs: Dict[str, str] = {}
    for header, seq in _read_fasta(path):
        m = FASTA_HEADER_RE.match(header)
        if not m:
            raise ValueError(f"Malformed FASTA header (sequences): {header}")
        uid = m.group("uid").upper()
        seqs[uid] = seq
    return seqs


def read_label_fasta(path: Path) -> list[LabelRecord]:
    """Parse from_publication.fasta into LabelRecord entries."""
    records: list[LabelRecord] = []
    for header, compact in _read_fasta(path):
        m = FASTA_HEADER_RE.match(header)
        if not m:
            raise ValueError(f"Malformed FASTA header (labels): {header}")
        uid = m.group("uid").upper()
        set_name = (m.group("set") or "").lower()
        val_str = (m.group("val") or "false").lower()
        is_validation = val_str in {"true", "1", "yes"}
        records.append(LabelRecord(uid, set_name, is_validation, compact))
    return records


# -----------------------------
# Label decoding
# -----------------------------
def _decode_compact_labels(
    compact: str,
    label_types: Sequence[str],
) -> torch.Tensor:
    """
    Decode a compact per-residue label string (digits '0'..'7') to a (L, K) multi-hot tensor.

    - compact: e.g., "000410..." where each char is one octal digit encoding 3 bits
    - label_types: order of channels to include, subset of DEFAULT_LABEL_TYPES
    """
    for lt in label_types:
        if lt not in BIT_INDEX_FOR_LABEL:
            raise ValueError(f"Unknown label type: {lt}. Valid: {list(BIT_INDEX_FOR_LABEL)}")

    L = len(compact)
    K = len(label_types)
    out = torch.zeros((L, K), dtype=torch.float32)

    # Vectorized over residue positions (still Python-side loop, but lean)
    bit_idxs = [BIT_INDEX_FOR_LABEL[lt] for lt in label_types]
    for i, ch in enumerate(compact):
        if ch < "0" or ch > "7":
            raise ValueError(f"Invalid label digit '{ch}' (expected '0'..'7')")
        val = ord(ch) - ord("0")  # 0..7
        # Fill channels in requested order
        for k, b in enumerate(bit_idxs):
            out[i, k] = float((val >> b) & 1)
    return out.int()


# -----------------------------
# Torch Dataset
# -----------------------------
class FlipDataset(Dataset):
    """
    Torch dataset returning DataElement with per-residue labels.

    Each sample:
      - protein_id: UniProt ID
      - length: sequence length
      - labels: Tensor[L, K] with channels in `label_types` order
      - seq: amino-acid sequence (string)
      - structure_path: local path to AlphaFold PDB
    """

    def __init__(
        self,
        df: pd.DataFrame,
        label_types: Sequence[str],
    ):
        self.df = df.reset_index(drop=True)
        self.label_types = tuple(label_types)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> DataElement:
        row = self.df.iloc[index]
        labels: torch.Tensor = row.residue_labels  # (L, K)
        return DataElement(
            protein_id=row.protein_id,
            length=len(row.seq),
            labels=labels,
            seq=row.seq,
            structure_path=str(row.structure_path),
        )


# -----------------------------
# Data module
# -----------------------------
class FlipModule:
    """
    Data module for FLIP-style dataset with:
      - sequences.fasta (UniProt headers, amino-acid sequence)
      - from_publication.fasta (UniProt + SET=... + VALIDATION=..., per-residue labels as digits '0'..'7')

    Flow:
      1) Read both FASTAs
      2) Merge on UniProt ID
      3) Download missing AlphaFold structures (via struct_template)
      4) Extract sequences from PDBs and enforce sequence identity with provided sequences
      5) Drop proteins lacking structures or mismatched sequences
      6) Decode per-residue labels to (L, K) tensors, optionally choosing subset of channels
    """

    def __init__(
        self,
        data_dir: str | Path,
        struct_template: str,
        *,
        seq_fasta_name: str = "sequences.fasta",
        labels_fasta_name: str = "from_publication.fasta",
        label_types: Sequence[str] = DEFAULT_LABEL_TYPES,
        num_workers: int = 16,
    ):
        self.data_dir = Path(data_dir)
        self.seq_fasta = self.data_dir / seq_fasta_name
        self.labels_fasta = self.data_dir / labels_fasta_name
        self.struct_template = struct_template
        self.num_workers = num_workers

        # Validate label_types
        for lt in label_types:
            if lt not in DEFAULT_LABEL_TYPES:
                raise ValueError(
                    f"Unknown label type '{lt}'. Choose from {list(DEFAULT_LABEL_TYPES)}"
                )
        self.label_types = tuple(label_types)
        self.df = self._get_flip_df()

    # ------------- Public API -------------
    def num_classes(self) -> int:
        return len(self.label_types)

    def get_dataset(self, split: Literal["train", "val", "test"]) -> Dataset:
        if split == "train":
            want = self.df.loc[lambda x: (x.set_name == "train") & (~x.is_validation)]
        elif split == "val":
            # By convention: validation examples are flagged as VALIDATION=True within the train set
            want = self.df.loc[lambda x: (x.set_name == "train") & (x.is_validation)]
        elif split == "test":
            want = self.df.loc[lambda x: x.set_name == "test"]
        else:
            raise ValueError(f"Unknown split: {split}")
        return FlipDataset(want, self.label_types)

    # ------------- Pipeline -------------
    def _get_flip_df(self) -> pd.DataFrame:
        # 1) Read FASTAs
        logger.info(f"Reading sequences from: {self.seq_fasta}")
        seqs = read_sequence_fasta(self.seq_fasta)

        logger.info(f"Reading labels from: {self.labels_fasta}")
        label_records = read_label_fasta(self.labels_fasta)

        labels_df = pd.DataFrame(
            [
                {
                    "uniprot_id": r.uniprot_id,
                    "set_name": r.set_name,
                    "is_validation": r.is_validation,
                    "labels_compact": r.labels_compact,
                }
                for r in label_records
            ]
        )

        seq_df = (
            pd.DataFrame({"uniprot_id": list(seqs.keys()), "seq": list(seqs.values())})
            .assign(length=lambda x: x.seq.str.len())
        )

        # 2) Merge on UniProt ID (inner)
        merged = labels_df.merge(seq_df, on="uniprot_id", how="inner")
        logger.info(
            f"Merged {len(labels_df)} label entries with {len(seq_df)} sequences -> {len(merged)}"
        )

        # Sanity check label lengths == sequence lengths
        same_len = merged.apply(lambda r: len(r.labels_compact) == len(r.seq), axis=1)
        if not same_len.all():
            n_bad = (~same_len).sum()
            raise ValueError(f"{n_bad} entries have label length != sequence length. Example: {merged.loc[~same_len].uniprot_id.head()}")

        # 3) Attach structure paths and download missing
        merged = merged.assign(
            structure_path=lambda x: x.uniprot_id.map(
                lambda uid: Path(self.struct_template % uid)
            )
        )

        file_paths = merged.structure_path.tolist()
        missing = [p for p in file_paths if not p.exists()]
        logger.info(f"downloading {len(missing)} / {len(file_paths)} missing AlphaFold PDBs")
        download_afdb_files(missing, num_workers=self.num_workers)

        # Remove entries with no structures
        has_struct = merged.structure_path.apply(lambda p: p.exists())
        if not has_struct.all():
            n_drop = (~has_struct).sum()
            logger.warning(f"Dropping {n_drop} entries with missing structures.")
            merged = merged.loc[has_struct].reset_index(drop=True)

        # 4) Parse AA sequences from PDBs (cached FASTA for speed)
        fasta_cache_path = self.data_dir / "flip.seq_from_pdb.fa"
        logger.info("Extracting sequences from PDBs to verify dataset sequences.")
        seq_from_pdb: Dict[str, str] = parse_seqs_from_pdbs(
            fasta_path=fasta_cache_path,
            file_paths=merged.structure_path.tolist(),
            uniprot_ids=merged.uniprot_id.tolist(),
            num_workers=self.num_workers,
        )

        # 5) Enforce exact sequence match with provided sequences
        merged = merged.assign(seq_pdb=lambda x: x.uniprot_id.map(seq_from_pdb))
        match = merged.apply(lambda r: r.seq == r.seq_pdb, axis=1)
        if not match.all():
            n_mismatch = (~match).sum()
            logger.warning(f"Dropping {n_mismatch} entries with sequence mismatches vs PDB.")
            merged = merged.loc[match].reset_index(drop=True)

        # 6) Decode per-residue labels to (L, K) tensors
        logger.info(f"Decoding per-residue labels to channels: {self.label_types}")
        merged = merged.assign(
            residue_labels=lambda x: [
                _decode_compact_labels(compact, self.label_types)
                for compact in tqdm(x.labels_compact.tolist(), desc="Decoding labels")
            ]
        )

        # final ordering
        merged = merged.rename(columns={"uniprot_id": "protein_id"})
        logger.info(f"Prepared {len(merged)} total entries after filtering.")
        return merged

