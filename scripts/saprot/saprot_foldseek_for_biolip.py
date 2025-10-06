import bz2
import logging
import re
import subprocess
import tempfile
from multiprocessing import Pool
from pathlib import Path

import fire
import pandas as pd
from tqdm import tqdm

from magneton.data.evaluations.biolip_dataset import BioLIP2Module

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# The steps we need to perform here are:
# - For each of train, val, test
# - Gather all the structure paths, run foldseek
# - For each protein's output in the foldseek output file
#   - Collect the structure tokens, which are only for unmasked positions
#   - Insert the placeholder 'd'token for all masked positions
# - Write these out to a FASTA file


def generate_raw_foldseek_output(
    df: pd.DataFrame,
    output_path: Path,
    num_workers: int = 32,
):
    num_pdbs = len(df.structure_path)
    logger.info(f"got {num_pdbs} pdb paths for FoldSeek tokens")

    # Temporary directory will be deleted automatically on exit
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create symlinks: protein_id.pdb -> actual structure file
        for _, row in df.iterrows():
            link_path = tmpdir_path / f"{row.protein_id}.pdb"
            # There are occasionally duplicates in these datasets
            if not link_path.exists():
                link_path.symlink_to(row.structure_path)

        # Run foldseek
        cmd = [
            "foldseek",
            "structureto3didescriptor",
            "--gpu",
            "1",
            "--threads",
            str(num_workers),
            "--chain-name-mode",
            "1",
            str(tmpdir_path),
            str(output_path),
        ]
        logger.info("Running FoldSeek: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)

    logger.info("FoldSeek completed and temporary directory cleaned up.")


def parse_foldseek_toks(
    df: pd.DataFrame,
    foldseek_output_path: Path,
    split: str,
) -> dict[str, str]:
    toks = {}
    with open(foldseek_output_path) as fh:
        for line in fh:
            parts = line.split("\t")
            name = parts[0].split()[0]
            name = re.sub(r'\.pdb_.*$', '', name)
            toks[name] = parts[2]

    parsed_toks = {}
    for _, row in tqdm(df.iterrows()):
        want_name = row.protein_id
        if want_name not in toks:
            logger.warning(f"{split} - {want_name}: not found")
            raw_toks = None
        else:
            raw_toks = toks[want_name]

            if len(raw_toks) != len(row.seq):
                if len(raw_toks) == len(row.seq) - 1:
                    raw_toks += "d"
                else:
                    logger.warning(
                        f"{split} - {want_name}: {len(raw_toks)} != {len(row.seq)}"
                    )
                    continue
        parsed_toks[row.protein_id] = raw_toks
    return parsed_toks


def run(
    biolip_path: str = "/weka/scratch/weka/kellislab/rcalef/data/magneton-data/evaluations/struct_token_bench",
    num_workers: int = 32,
):
    biolip_path = Path(biolip_path)
    for task in ["binding", "catalytic"]:
        module = BioLIP2Module(
            data_dir=biolip_path,
            task=task,
        )
        for split in ["val", "train", "test"]:
            df = module.get_dataset(split).dataset

            split_dir = biolip_path.parent / f"biolip_{task}" / split
            split_dir.mkdir(parents=True, exist_ok=True)

            foldseek_tokens_path = split_dir / "foldseek_tokens.fa.bz2"
            foldseek_output_path = (
                foldseek_tokens_path.parent / f"{foldseek_tokens_path.stem}.tsv"
            )
            foldseek_output_dbtype_path = (
                foldseek_tokens_path.parent / f"{foldseek_tokens_path.stem}.tsv.dbtype"
            )

            generate_raw_foldseek_output(df, foldseek_output_path, num_workers=num_workers)
            toks_by_id = parse_foldseek_toks(
                df,
                foldseek_output_path,
                split,
            )
            with bz2.open(foldseek_tokens_path, "wt") as out_fh:
                for protein_id, full_toks in toks_by_id.items():
                    out_fh.write(f">{protein_id}\n{full_toks.lower()}\n")

            foldseek_output_path.unlink()
            foldseek_output_dbtype_path.unlink()


if __name__ == "__main__":
    fire.Fire(run)
