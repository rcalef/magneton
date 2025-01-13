import bz2
import inspect
import logging
import os
import pickle
import re
import sys

from functools import partial
from multiprocessing import Pool, get_logger
from operator import itemgetter
from typing import Callable, Generator, List, Optional, Tuple

from pysam import FastaFile
from tqdm import tqdm

from magneton.types import Protein

def parse_from_pkl(
    input_path: str,
    compression: Optional[str] = None,
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
    prefix: str = "parsed_proteins",
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
    prefix: str = "parsed_proteins",
) -> Generator[Tuple[Protein, str], None, None]:
    all_files = get_sorted_files(dir, prefix)

    for fn in all_files:
        for prot in parse_from_pkl(os.path.join(dir, fn), compression="bz2"):
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

                print(f"completed file {curr_file_num}, starting file {curr_file_num+1}")
                output_path = os.path.join(output_dir, f"{prefix}.{curr_file_num}.pkl.bz2")
                output_fh = bz2.open(output_path, "wb")

    output_fh.close()

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

    all_files = [os.path.join(shard_dir, x[1]) for x in get_sorted_files(shard_dir, prefix)]

    with Pool(nprocs) as p:
        return list(tqdm(p.imap(func, all_files), total=len(all_files)))

