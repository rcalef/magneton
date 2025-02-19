import os

from bisect import bisect
from typing import Generator

import pandas as pd

from torch.utils.data import Dataset, IterableDataset

from magneton.io.internal import parse_from_dir, parse_from_pkl
from magneton.types import Protein


class InMemoryProteinDataset(Dataset):
    def __init__(
        self,
        proteins: Generator[Protein, None, None],
    ):
        super().__init__()

        self.proteins = list(proteins)

    def __iter__(self) -> Generator[Protein, None, None]:
        return iter(self.proteins)

    def __len__(self) -> int:
        return len(self.proteins)

    def __getitem__(self, idx: int) -> Protein:
        return self.proteins[idx]

    def fetch_protein(self, uniprot_id: str) -> Protein:
        # NOTE: not optimized for random access, could alleviate this in the future with
        for prot in self.proteins:
            if prot.uniprot_id == uniprot_id:
                return prot
        raise ValueError(f"{uniprot_id} not found")


class ShardedProteinDataset(IterableDataset):
    def __init__(
        self,
        input_path: str,
        compression: str = "bz2",
        prefix: str = "sharded_proteins",
    ):
        super().__init__()
        self.input_path = input_path
        self.compression = compression
        self.prefix = prefix

        assert os.path.isdir(input_path)

        self._load_index()

    def _load_index(self):
        index = pd.read_table(
            os.path.join(self.input_path, "index.tsv"),
            names=["file_num", "index_entry", "file_len"],
        )
        self.index_entries = index.index_entry.tolist()
        self.length = index.file_len.sum()

    def __iter__(self) -> Generator[Protein, None, None]:
        yield from parse_from_dir(
            self.input_path, self.prefix, compression=self.compression
        )

    def __len__(self) -> int:
        return self.length

    def fetch_protein(self, uniprot_id: str) -> Protein:
        # This gives the first index entry that's greater than `uniprot_id`,
        # we want the last index entry that's less than, hence the -1 below.
        index = bisect(self.index_entries, uniprot_id)
        if index == 0:
            raise ValueError(f"{uniprot_id} not found in {self.input_path}")
        index -= 1

        for prot in parse_from_pkl(
            os.path.join(
                self.input_path,
                f"{self.prefix}.{index}.pkl.{self.compression}",
            ),
            self.compression,
        ):
            if prot.uniprot_id == uniprot_id:
                return prot
        raise ValueError(f"{uniprot_id} not found in {self.input_path} index {index}")


def get_protein_dataset(
    input_path: str,
    compression: str = "bz2",
    in_memory: bool = False,
    prefix: str = "sharded_proteins",
) -> InMemoryProteinDataset | ShardedProteinDataset:
    if os.path.isdir(input_path):
        if in_memory:
            return InMemoryProteinDataset(
                parse_from_dir(input_path, prefix=prefix, compression=compression)
            )
        else:
            return ShardedProteinDataset(
                input_path,
                compression=compression,
                prefix=prefix,
            )
    elif ".pkl" in input_path:
        return InMemoryProteinDataset(
            parse_from_pkl(input_path, compression=compression)
        )
    else:
        raise ValueError(f"expected dir or pickle file, got: {input_path}")
