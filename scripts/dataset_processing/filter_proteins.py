import logging
import os
from functools import partial
from itertools import chain
from typing import List

import fire
from pysam import FastaFile

from magneton.core_types import Protein
from magneton.io.internal import (
    parse_from_json,
    process_sharded_proteins,
    shard_proteins,
)


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
    for prot in parse_from_json(input_path, compression="gz"):
        if prot.uniprot_id in want_ids:
            filtered_proteins.append(prot)
        tot += 1

    logger.info(
        f"Retained {len(filtered_proteins)} / {tot} proteins from {os.path.basename(input_path)}"
    )
    return filtered_proteins


def filter_dir(
    input_dir: str,
    output_dir: str,
    filter_path: str,
    output_prefix: str = "sharded_proteins",
    prots_per_shard: int = 10000,
    nprocs: int = 16,
) -> List[Protein]:
    filter_func = partial(filter_proteins, filter_path=filter_path)
    filtered_proteins = process_sharded_proteins(
        input_dir,
        filter_func,
        nprocs=nprocs,
    )
    shard_proteins(
        chain(*filtered_proteins),
        output_dir=output_dir,
        prefix=output_prefix,
        prots_per_file=prots_per_shard,
    )


if __name__ == "__main__":
    fire.Fire(filter_dir)
