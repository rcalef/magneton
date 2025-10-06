import bz2
import logging
import os
import pickle

from typing import Tuple

from functools import partial

import fire

from magneton.io.internal import (
    parse_from_pkl,
    process_sharded_proteins,
)
from magneton.io.mmcif import mmcif_to_secondary_structs
from magneton.core_types import Protein


def add_secondary_structs_to_protein(
    prot: Protein,
    path_tmpl: str,
) -> Tuple[Protein, bool]:
    path = path_tmpl % prot.uniprot_id
    # Some proteins don't have structures from PDB or AlphaFoldDB
    if not os.path.exists(path):
        return prot, False

    try:
        prot.secondary_structs = mmcif_to_secondary_structs(path, expected_len=prot.length)
    except Exception as e:
        print(e)
        return prot, False
    return prot, True


def add_ss_to_interpro_pkl(
    pkl_path: str,
    outdir: str,
    prefix: str,
    cif_tmpl: str,
    logger: logging.Logger = logging.getLogger(__name__),
):
    num_kept = 0
    tot = 0
    outpath = os.path.join(
        outdir, os.path.basename(pkl_path.replace(prefix, f"{prefix}.with_ss"))
    )

    first_kept = None
    logger.info(f"{os.path.basename(pkl_path)}: begin")
    with bz2.open(outpath, "wb") as fh:
        for prot in parse_from_pkl(pkl_path, compression="bz2"):
            tot += 1
            try:
                prot_with_ss, has_struct = add_secondary_structs_to_protein(
                    prot, cif_tmpl
                )
            except Exception as e:
                logger.info(f"{os.path.basename(pkl_path)}: {prot.uniprot_id} failed")
                raise e
            if has_struct:
                num_kept += 1
                pickle.dump(prot_with_ss, fh)
                if first_kept is None:
                    first_kept = prot.uniprot_id
    logger.info(f"{os.path.basename(pkl_path)}: found {tot-num_kept} / {tot} missing")
    return first_kept, num_kept


def add_ss_to_interpro_sharded(
    dir_path: str,
    outdir: str,
    cif_tmpl: str,
    nprocs: int = 32,
    prefix: str = "sharded_proteins",
):
    new_index_entries = process_sharded_proteins(
        dir_path,
        partial(
            add_ss_to_interpro_pkl, prefix=prefix, outdir=outdir, cif_tmpl=cif_tmpl
        ),
        nprocs=nprocs,
        prefix=prefix,
    )
    with open(os.path.join(outdir, "index.tsv"), "w") as fh:
        for file_num, (new_index_entry, new_index_size) in enumerate(new_index_entries):
            fh.write(f"{file_num}\t{new_index_entry}\t{new_index_size}\n")


if __name__ == "__main__":
    fire.Fire(add_ss_to_interpro_sharded)
