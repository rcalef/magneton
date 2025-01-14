import logging
import os

from typing import Dict, Tuple
from functools import partial

import fire
import pandas as pd

from magneton.io.internal import (
    parse_from_pkl,
    process_sharded_proteins,
)
from magneton.summary_stats import (
    calc_summaries,
    merge_summaries,
)

def summary_with_log(
    path: str,
    logger: logging.Logger,
    labels_path: str = "/weka/scratch/weka/kellislab/rcalef/data/interpro/102.0/label_sets/",
) -> Tuple[pd.DataFrame, Dict]:
    logger.info(f"{os.path.basename(path)}: starting summary calculation")
    summaries, substructure_metrics = calc_summaries(parse_from_pkl(path, compression="bz2"), labels_path=labels_path)
    logger.info(f"{os.path.basename(path)}: finished summary calculation")
    return summaries, substructure_metrics



def calc_and_write_summaries(
    dir_path: str,
    outdir: str,
    labels_path: str = "/weka/scratch/weka/kellislab/rcalef/data/interpro/102.0/label_sets/",
    nprocs: int = 32,
    prefix: str = "sharded_proteins",
):
    results = process_sharded_proteins(
        dir_path,
        partial(summary_with_log, labels_path=labels_path),
        nprocs=nprocs,
        prefix=prefix,
    )
    prot_summaries, struct_summaries = zip(*results)
    merged_prot_summaries, merged_struct_summaries = merge_summaries(prot_summaries, struct_summaries)
    with open(os.path.join(outdir, "protein_summaries.tsv"), "w") as fh:
        merged_prot_summaries.to_csv(fh, sep="\t", index=False)
    for interpro_type, metrics in merged_struct_summaries.items():
        with open(os.path.join(outdir, f"{interpro_type}_summaries.tsv"), "w") as fh:
            metrics.to_csv(fh, sep="\t", index=False)


if __name__ == "__main__":
    fire.Fire(calc_and_write_summaries)
