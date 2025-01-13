import bz2
import logging
import os
import pickle
import sys

from functools import partial
from multiprocessing import Pool, get_logger
from typing import List

import fire

from pysam import FastaFile

from magneton.io.internal import get_sorted_files, parse_from_pkl
from magneton.types import Protein


def filter_proteins(
    input_path: str,
    filter_fai_path: str,
) -> List[Protein]:
    logger = get_logger()

    fa = FastaFile(filter_fai_path)
    want_ids = {x for x in fa.references}

    logger.info(f"Filtering file {os.path.basename(input_path)}")
    filtered_proteins = []
    tot = 0
    for prot in parse_from_pkl(input_path, compression="bz2"):
        if prot.kb_id in want_ids:
            filtered_proteins.append(prot)
        tot += 1

    logger.info(
        f"Retained {len(filtered_proteins)} / {tot} proteins from {os.path.basename(input_path)}"
    )
    return filtered_proteins


def filter_dir(
    dir_path: str,
    output_path: str,
    filter_fai_path: str,
    nproc: int = 16,
) -> List[Protein]:
    logger = get_logger()
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    all_files = [os.path.join(dir_path, x[1]) for x in get_sorted_files(dir_path)]
    logger.info(f"Got {len(all_files)} files to filter")

    filter_func = partial(filter_proteins, filter_fai_path=filter_fai_path)
    with Pool(nproc) as p:
        filtered_proteins = p.map(filter_func, all_files)

    with bz2.open(output_path, "wb") as fh:
        for subres in filtered_proteins:
            for prot in subres:
                pickle.dump(prot, fh)


if __name__ == "__main__":
    fire.Fire(filter_dir)
