from bisect import bisect
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Generator

import pandas as pd
from torch.utils.data import Dataset, IterableDataset

from magneton.core_types import Protein
from magneton.io.internal import (
    filter_proteins,
    get_sorted_files,
    parse_from_dir,
    parse_from_json,
)

from .substructure_parsers import BaseSubstructureParser


class InMemoryProteinDataset(Dataset):
    """Dataset of Protein objects stored in memory."""

    def __init__(
        self,
        proteins: list[Protein],
    ):
        super().__init__()

        self.proteins = proteins

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
    """Dataset that yields protein objects from sharded files."""

    def __init__(
        self,
        input_path: str,
        compression: str = "gz",
        prefix: str = "sharded_proteins",
    ):
        super().__init__()
        self.input_path = Path(input_path)
        self.compression = compression
        self.prefix = prefix

        assert self.input_path.exists() and self.input_path.is_dir()
        all_files = get_sorted_files(input_path, prefix)
        if len(all_files) == 0:
            raise ValueError(f"no files found in {input_path} with prefix {prefix}")

        self._load_index()

    def _load_index(self):
        index = pd.read_table(
            self.input_path / "index.tsv",
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

        for prot in parse_from_json(
            self.input_path / f"{self.prefix}.{index}.jsonl.{self.compression}",
            self.compression,
        ):
            if prot.uniprot_id == uniprot_id:
                return prot
        raise ValueError(f"{uniprot_id} not found in {self.input_path} index {index}")


def protein_has_substructs(
    prot: Protein,
    want_subtype_parser: BaseSubstructureParser,
) -> bool:
    """
    Check if any of the substructure entries are of the desired type.
    """
    parsed = want_subtype_parser.parse(prot)
    return len(parsed) > 0


def passthrough_filter_func(prot: Protein) -> bool:
    return True


def protein_in_subset(prot: Protein, subset: set[str]) -> bool:
    return prot.uniprot_id in subset


def check_filters(prot: Protein, funcs: list[Callable[[Protein], bool]]) -> bool:
    for func in funcs:
        if not func(prot):
            return False
    return True


def get_protein_dataset(
    input_path: str | Path,
    compression: str = "gz",
    in_memory: bool = False,
    prefix: str = "sharded_proteins",
    nprocs: int = 32,
    want_subtype_parser: BaseSubstructureParser | None = None,
    want_subset: list[str] | None = None,
) -> InMemoryProteinDataset | ShardedProteinDataset:
    """Create a Protein dataset with the desired configurations."""
    filters = []
    if want_subtype_parser is not None:
        filters.append(
            partial(protein_has_substructs, want_subtype_parser=want_subtype_parser)
        )
    if want_subset is not None:
        filters.append(partial(protein_in_subset, subset=set(want_subset)))
    if len(filters) > 0:
        filter_func = partial(check_filters, funcs=filters)
    else:
        filter_func = passthrough_filter_func

    if isinstance(input_path, str):
        input_path = Path(input_path)
    if not input_path.exists():
        raise ValueError(f"path not found: {str(input_path)}")

    if input_path.is_dir():
        if in_memory:
            return InMemoryProteinDataset(
                list(
                    filter_proteins(
                        input_path,
                        prefix=prefix,
                        compression=compression,
                        nprocs=nprocs,
                        filter_func=filter_func,
                    )
                )
            )
        else:
            return ShardedProteinDataset(
                input_path,
                compression=compression,
                prefix=prefix,
            )
    elif ".jsonl" in str(input_path):
        return InMemoryProteinDataset(
            [
                x
                for x in parse_from_json(input_path, compression=compression)
                if filter_func(x)
            ]
        )
    else:
        raise ValueError(f"expected dir or JSONL file, got: {str(input_path)}")
