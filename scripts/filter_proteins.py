import bz2
import logging
import os
import pickle

from functools import partial
from multiprocessing import Pool, get_logger
from typing import List

import fire

from pysam import FastaFile

<<<<<<< Updated upstream
from magneton.io.internal import parse_from_pkl, process_sharded_proteins
=======
from magneton.io.internal import get_sorted_files, parse_from_pkl
>>>>>>> Stashed changes
from magneton.types import Protein


def filter_proteins(
    input_path: str,
    filter_path: str,
    logger: logging.Logger,
) -> List[Protein]:
    if filter_path.endswith(".fasta.bgz"):
        fa = FastaFile(filter_path)
        want_ids = {x.split("|")[1] for x in fa.references}
    elif filter_path.endswith(".tsv"):
        want_ids = set()
        with open(filter_path, "r") as fh:
            for line in fh:
                want_ids.add(line.strip())
    else:
        raise ValueError(f"Unknown filter file type: {filter_path}")

    logger.info(f"Filtering file {os.path.basename(input_path)}")
    filtered_proteins = []
    tot = 0
    for prot in parse_from_pkl(input_path, compression="bz2"):
        if prot.uniprot_id in want_ids:
            filtered_proteins.append(prot)
        tot += 1

    logger.info(
        f"Retained {len(filtered_proteins)} / {tot} proteins from {os.path.basename(input_path)}"
    )
    return filtered_proteins


def filter_dir(
    dir_path: str,
    output_path: str,
    filter_path: str,
    nprocs: int = 16,
) -> List[Protein]:
    filter_func = partial(filter_proteins, filter_path=filter_path)
    filtered_proteins = process_sharded_proteins(
        dir_path,
        filter_func,
        nprocs=nprocs,
    )

    with bz2.open(output_path, "wb") as fh:
        for subres in filtered_proteins:
            for prot in subres:
                pickle.dump(prot, fh)


if __name__ == "__main__":
    fire.Fire(filter_dir)
