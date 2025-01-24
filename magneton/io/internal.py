import bz2
import inspect
import logging
import os
import pickle
import re
import sys

from bisect import bisect
from collections.abc import Callable
from functools import partial
from itertools import chain
from multiprocessing import Pool, get_logger
from operator import itemgetter
from typing import Generator, List, Optional, Tuple

from pysam import FastaFile
from tqdm import tqdm

from magneton.types import Protein


class ProteinDataset:
    def __init__(
        self,
        input_path: str,
        compression: str = "bz2",
        fasta_path: Optional[str] = None,
        prefix: str = "sharded_proteins",
    ):
        self.input_path = input_path
        self.compression = compression
        self.fasta_path = fasta_path
        self.prefix = prefix

        if os.path.isdir(input_path):
            self.backing = "dir"
            self._load_index()
        elif ".pkl" in input_path:
            self.backing = "pkl"
        else:
            raise ValueError(f"expected dir or pickle file, got: {input_path}")

    def __iter__(self) -> Generator[Protein, None, None]:
        if self.backing == "dir":
            yield from parse_from_dir(
                self.input_path, self.prefix, compression=self.compression
            )
        elif self.backing == "pkl":
            yield from parse_from_pkl(self.input_path, compression=self.compression)

    def _load_index(self):
        index_path = os.path.join(self.input_path, "index.txt")
        index_entries = []
        with open(index_path, "r") as index_fh:
            for line in index_fh:
                index_entries.append(line.strip().split("\t")[1])
        self.index_entries = index_entries

    def fetch_protein(self, uniprot_id: str) -> Protein:
        assert self.backing == "dir"
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


def parse_from_pkl(
    input_path: str,
    compression: Optional[str] = "bz2",
) -> Generator[Protein, None, None]:
    if compression is None:
        open_fn = open
    elif compression == "bz2":
        open_fn = bz2.open
    else:
        raise ValueError(f"unknown compression: {compression}")
    with open_fn(input_path, "rb") as fh:
        while True:
            try:
                yield pickle.load(fh)
            except EOFError as e:
                break
            except Exception as e:
                raise e


def parse_from_pkl_w_fasta(
    input_path: str,
    fasta_path: str,
) -> Generator[Tuple[Protein, str], None, None]:
    fa = FastaFile(fasta_path)
    for prot in parse_from_pkl(input_path):
        if prot.kb_id in fa:
            yield (prot, fa[prot.kb_id])


def get_sorted_files(
    dir: str,
    prefix: str = "sharded_proteins",
) -> List[Tuple[int, str]]:
    pat = re.compile(f"{prefix}.(\d+).pkl.bz2")
    all_files = []
    for fn in os.listdir(dir):
        res = pat.match(fn)
        if res:
            all_files.append((int(res.group(1)), fn))
    return sorted(all_files, key=itemgetter(0))


def parse_from_dir(
    dir: str,
    prefix: str = "sharded_proteins",
    compression: str = "bz2",
) -> Generator[Tuple[Protein, str], None, None]:
    all_files = get_sorted_files(dir, prefix)
    if len(all_files) == 0:
        raise ValueError(f"no files found in {dir} with prefix {prefix}")

    for _, fn in all_files:
        for prot in parse_from_pkl(os.path.join(dir, fn), compression=compression):
            yield prot


def shard_proteins(
    input_iter: Generator[Protein, None, None],
    output_dir: str,
    prefix: str = "sharded_proteins",
    prots_per_file: int = 500000,
):
    """Transform generator of proteins into sharded files"""
    index_path = os.path.join(output_dir, "index.txt")

    with open(index_path, "w") as index_fh:
        curr_file_num = 0
        prev_id = "0"
        curr_file_prots = 0

        output_path = os.path.join(output_dir, f"{prefix}.{curr_file_num}.pkl.bz2")
        output_fh = bz2.open(output_path, "wb")
        for prot in input_iter:
            assert prev_id < prot.uniprot_id, f"{prev_id} !< {prot.uniprot_id}"
            prev_id = prot.uniprot_id

            if curr_file_prots == 0:
                index_fh.write(f"{curr_file_num}\t{prot.uniprot_id}\n")

            pickle.dump(prot, output_fh)
            curr_file_prots += 1

            if curr_file_prots == prots_per_file:
                output_fh.close()
                curr_file_num += 1
                curr_file_prots = 0

                print(
                    f"completed file {curr_file_num}, starting file {curr_file_num+1}"
                )
                output_path = os.path.join(
                    output_dir, f"{prefix}.{curr_file_num}.pkl.bz2"
                )
                output_fh = bz2.open(output_path, "wb")

    output_fh.close()

def _filter_protein_file(
    input_path: str,
    filter_func: Callable[[Protein], bool],
) -> List[Protein]:
    return [prot for prot in parse_from_pkl(input_path) if filter_func(prot)]

def filter_proteins(
    shard_dir: str,
    filter_func: Callable[[Protein], bool],
    nprocs: int = 32,
    prefix: str = "sharded_proteins",
) -> List[Protein]:
    filter_func=partial(_filter_protein_file, filter_func=filter_func)
    filtered_lists = process_sharded_proteins(
        shard_dir=shard_dir,
        func=filter_func,
        nprocs=nprocs,
        prefix=prefix,
    )
    return list(chain.from_iterable(filtered_lists))

def process_sharded_proteins(
    shard_dir: str,
    func: Callable,
    nprocs: int = 32,
    prefix: str = "sharded_proteins",
) -> List:
    """Run a function on each shard in a directory"""

    # Check if func takes a `logger` argument
    if "logger" in inspect.signature(func).parameters:
        logger = get_logger()
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        func = partial(func, logger=logger)

    all_files = [
        os.path.join(shard_dir, x[1]) for x in get_sorted_files(shard_dir, prefix)
    ]

    with Pool(nprocs) as p:
        return list(tqdm(p.imap(func, all_files), total=len(all_files)))
