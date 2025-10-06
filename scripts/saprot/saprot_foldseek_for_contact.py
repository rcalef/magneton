import bz2
import logging
import subprocess
import tempfile
from multiprocessing import Pool
from pathlib import Path

import fire
import pandas as pd
from tqdm import tqdm

from magneton.data.evaluations import ContactPredictionModule

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
            link_path = tmpdir_path / f"{row.pdb_id}.pdb"
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
            toks[name] = parts[2]

    parsed_toks = {}
    for _, row in tqdm(df.iterrows()):
        if split == "val":
            want_name = f"{row.pdb_id}.pdb_{row.chain}"
        elif split == "train":
            want_name = f"{row.pdb_id}.pdb_{row.chain}"
        else:
            want_name = f"{row.pdb_id}.pdb_"

        if want_name not in toks:
            logger.warning(f"{split} - {want_name}: not found")
            raw_toks = None
        else:
            raw_toks = toks[want_name]

            if len(raw_toks) != len(row.valid_seq):
                logger.warning(
                    f"{split} - {want_name}: {len(raw_toks)} != {len(row.valid_seq)}"
                )

        invalid_pos = (row.labels == -1).all(dim=1)
        full_toks = []
        idx = 0
        for invalid in invalid_pos:
            if raw_toks is None or invalid or idx == len(raw_toks):
                this_tok = "d"
            else:
                this_tok = raw_toks[idx]
                idx += 1
            full_toks.append(this_tok)

        full_toks = "".join(full_toks)
        if len(full_toks) != len(row.seq):
            raise ValueError(
                f"{split} - {want_name}: {len(full_toks)} != {len(row.seq)}"
            )
        parsed_toks[row.protein_id] = full_toks
    return parsed_toks


def run(
    contact_data_path: str = "/weka/scratch/weka/kellislab/rcalef/data/magneton-data/evaluations/saprot_processed/Contact",
    num_workers: int = 32,
):
    contact_data_path = Path(contact_data_path)
    module = ContactPredictionModule(
        data_dir=contact_data_path,
    )
    for split in ["val", "train", "test"]:
        df = module._prepare_data(split)

        split_dir = contact_data_path / split
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
