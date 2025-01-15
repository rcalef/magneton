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
from magneton.custom_types import Protein


def add_secondary_structs_to_protein(
    prot: Protein,
    path_tmpl: str,
) -> Tuple[Protein, bool]:
    path = path_tmpl % prot.uniprot_id
    # Some proteins don't have structures from PDB or AlphaFoldDB
    if not os.path.exists(path):
        return prot, False

    prot.secondary_structs = mmcif_to_secondary_structs(path)
    return prot, True


def add_ss_to_interpro_pkl(
    pkl_path: str,
    outdir: str,
    cif_tmpl: str,
    logger: logging.Logger = logging.getLogger(__name__),
):
    num_missing = 0
    tot = 0
    outpath = os.path.join(outdir, os.path.basename(pkl_path.replace(".pkl.bz2", ".with_ss.pkl.bz2")))

    logger.info(f"{os.path.basename(pkl_path)}: begin")
    with bz2.open(outpath, "wb") as fh:
        for prot in parse_from_pkl(pkl_path, compression="bz2"):
            tot += 1
            try:
                prot_with_ss, has_struct = add_secondary_structs_to_protein(prot, cif_tmpl)
            except Exception as e:
                logger.info(f"{os.path.basename(pkl_path)}: {prot.uniprot_id} failed")
                raise e
            if has_struct:
                pickle.dump(prot_with_ss, fh)
            else:
                num_missing += 1
    logger.info(f"{os.path.basename(pkl_path)}: found {num_missing} / {tot} missing")


def add_ss_to_interpro_sharded(
    dir_path: str,
    outdir: str,
    cif_tmpl: str,
    nprocs: int = 32,
    prefix: str = "sharded_proteins",
):
    process_sharded_proteins(
        dir_path,
        partial(add_ss_to_interpro_pkl, outdir=outdir, cif_tmpl=cif_tmpl),
        nprocs=nprocs,
        prefix=prefix,
    )


if __name__ == "__main__":
    fire.Fire(add_ss_to_interpro_sharded)
