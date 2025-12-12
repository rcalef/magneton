import gzip
import json
import inspect
import logging
import os
import re
import sys

from collections.abc import Callable
from functools import partial
from itertools import chain
from multiprocessing import Pool, get_logger
from operator import itemgetter
from typing import Generator, List, Optional, Tuple

import pandas as pd

from pysam import FastaFile
from tqdm import tqdm

from magneton.core_types import Protein


def parse_from_json(
    input_path: str,
    compression: Optional[str] = "gz",
) -> Generator[Protein, None, None]:
    if compression is None:
        open_fn = open
    elif compression == "gz":
        open_fn = gzip.open
    else:
        raise ValueError(f"unknown compression: {compression}")
    with open_fn(input_path, "r") as fh:
        for line in fh:
            try:
                prot = Protein.fromJSON(json.loads(line))
                yield prot
            except Exception as e:
                e.add_note(f"file {input_path}, line: {line}")
                raise e


def parse_from_json_w_fasta(
    input_path: str,
    fasta_path: str,
) -> Generator[Tuple[Protein, str], None, None]:
    fa = FastaFile(fasta_path)
    for prot in parse_from_json(input_path):
        if prot.kb_id in fa:
            yield (prot, fa[prot.kb_id])


def get_sorted_files(
    dir: str,
    prefix: str = "sharded_proteins",
) -> List[Tuple[int, str]]:
    pat = re.compile(f"{prefix}.(\d+).jsonl.gz")
    all_files = []
    for fn in os.listdir(dir):
        res = pat.match(fn)
        if res:
            all_files.append((int(res.group(1)), fn))
    return sorted(all_files, key=itemgetter(0))


def parse_from_dir(
    dir: str,
    prefix: str = "sharded_proteins",
    compression: str = "gz",
    filter_func: Optional[Callable[[Protein], bool]] = None,
) -> Generator[Tuple[Protein, str], None, None]:
    all_files = get_sorted_files(dir, prefix)
    if len(all_files) == 0:
        raise ValueError(f"no files found in {dir} with prefix {prefix}")

    for _, fn in all_files:
        for prot in parse_from_json(os.path.join(dir, fn), compression=compression):
            if filter_func is not None and not filter_func(prot):
                continue
            yield prot


def shard_proteins(
    input_iter: Generator[Protein, None, None],
    output_dir: str,
    prefix: str = "sharded_proteins",
    prots_per_file: int = 500000,
):
    """Transform generator of proteins into sharded files"""
    index_entries = []
    file_lens = []

    prev_id = "0"
    curr_file_prots = 0
    curr_file_num = 0

    output_path = os.path.join(output_dir, f"{prefix}.{curr_file_num}.jsonl.gz")
    output_fh = gzip.open(output_path, "wt")
    for prot in input_iter:
        assert prev_id < prot.uniprot_id, f"{prev_id} !< {prot.uniprot_id}"
        prev_id = prot.uniprot_id

        if curr_file_prots == 0:
            index_entries.append(prot.uniprot_id)

        print(prot.toJSON(), file=output_fh)
        curr_file_prots += 1

        if curr_file_prots == prots_per_file:
            file_lens.append(curr_file_prots)

            curr_file_prots = 0
            curr_file_num += 1

            print(f"completed file {curr_file_num}, starting file {curr_file_num + 1}")
            output_fh.close()
            output_path = os.path.join(output_dir, f"{prefix}.{curr_file_num}.jsonl.gz")
            output_fh = gzip.open(output_path, "wt")

    index = pd.DataFrame(
        {
            "file_num": range(len(index_entries)),
            "index_entry": index_entries,
            "file_len": file_lens + [curr_file_prots],
        }
    )
    index.to_csv(
        os.path.join(output_dir, "index.tsv"), sep="\t", index=False, header=False
    )


def _filter_protein_file(
    input_path: str,
    filter_func: Callable[[Protein], bool],
    compression: Optional[str] = "gz",
) -> List[Protein]:
    return [
        prot
        for prot in parse_from_json(input_path, compression=compression)
        if filter_func(prot)
    ]


def filter_proteins(
    shard_dir: str,
    filter_func: Callable[[Protein], bool],
    prefix: str = "sharded_proteins",
    compression: Optional[str] = "gz",
    nprocs: int = 32,
) -> List[Protein]:
    filter_func = partial(
        _filter_protein_file,
        filter_func=filter_func,
        compression=compression,
    )
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
    message: str = "processing proteins",
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
    if len(all_files) == 0:
        raise RuntimeError(
            f"No files matching prefix '{prefix}' found in shard_dir '{shard_dir}'"
        )

    with Pool(nprocs) as p:
        return list(tqdm(p.imap(func, all_files), total=len(all_files), desc=message))


def passthrough(prot: Protein) -> bool:
    return True


def parse_from_dir_parallel(
    dir: str,
    prefix: str = "sharded_proteins",
    compression: str = "gz",
    filter_func: Optional[Callable[[Protein], bool]] = None,
    nprocs: int = 32,
) -> Generator[Tuple[Protein, str], None, None]:
    if filter_func is None:
        filter_func = passthrough
    return filter_proteins(
        shard_dir=dir,
        filter_func=filter_func,
        prefix=prefix,
        compression=compression,
        nprocs=nprocs,
    )
