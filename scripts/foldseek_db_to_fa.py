import argparse
import gzip

from functools import partial
from multiprocessing import Pool

import numpy as np

from Bio.PDB import PDBParser
from tqdm import tqdm



def extract_plddt(pdb_path: str) -> np.ndarray:
    """
    Extract plddt scores from pdb file.
    Args:
        pdb_path: Path to pdb file.

    Returns:
        plddts: plddt scores.
    """

    # Initialize parser
    parser = PDBParser()

    structure = parser.get_structure('protein', pdb_path)
    model = structure[0]
    chain = model["A"]

    # Extract plddt scores
    plddts = []
    for residue in chain:
        residue_plddts = []
        for atom in residue:
            plddt = atom.get_bfactor()
            residue_plddts.append(plddt)

        plddts.append(np.mean(residue_plddts))

    plddts = np.array(plddts)
    return plddts

def mask_foldseek(
    pdb_path: str,
    foldseek_seq: str,
    plddt_threshold: float = 70,
) -> str:
    plddts = extract_plddt(pdb_path)
    assert len(plddts) == len(foldseek_seq), f"Length mismatch: {len(plddts)} != {len(foldseek_seq)}"

    # Mask regions with plddt < threshold
    indices = np.where(plddts < plddt_threshold)[0]
    np_seq = np.array(list(foldseek_seq))
    np_seq[indices] = "#"
    return "".join(np_seq)

def run_one_job(
    line: str,
    pdb_dir: str,
) -> tuple[str, str]:
    protein_id, aa_seq, foldseek_seq = line.strip().split("\t")[:3]
    protein_id = protein_id.split()[0]
    uniprot_id = protein_id.strip().split("-")[1]

    fn = protein_id.replace("_A", "")
    full_path = f"{pdb_dir}/{fn}"
    masked_seq = mask_foldseek(full_path, foldseek_seq)

    return (uniprot_id, masked_seq)

def run_conversion(
    foldseek_path: str,
    output_path: str,
    pdb_dir: str,
    total: int | None = None,
    nprocs: int = 32,
):
    process_func = partial(run_one_job, pdb_dir=pdb_dir)
    with gzip.open(foldseek_path, "rt") as fh, Pool(nprocs) as p:
        results = list(tqdm(p.imap_unordered(process_func, fh), total=total))

    with gzip.open(output_path, "wt") as out_fh:
        for uniprot_id, masked_seq in results:
            out_fh.write(f">{uniprot_id}\n{masked_seq.lower()}\n")

# def run_conversion(
#     foldseek_path: str,
#     output_path: str,
#     pdb_dir: str,
#     total: int | None = None,
# ):
#     with gzip.open(foldseek_path, "rt") as fh, gzip.open(output_path, "wt") as out_fh:
#         for i, line in enumerate(tqdm(fh, total=total)):
#             protein_id, aa_seq, foldseek_seq = line.strip().split("\t")[:3]
#             protein_id = protein_id.split()[0]
#             uniprot_id = protein_id.strip().split("-")[1]

#             fn = protein_id.replace("_A", "")
#             full_path = f"{pdb_dir}/{fn}"
#             masked_seq = mask_foldseek(full_path, foldseek_seq)

#             out_fh.write(f">{uniprot_id}\n{masked_seq.lower()}\n")
#             if i == 100:
#                 break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Foldseek output to FASTA format."
    )
    parser.add_argument(
        "foldseek_path",
        type=str,
        help="Path to the input foldseek file (gzipped).",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to the output FASTA file (gzipped).",
    )
    parser.add_argument(
        "pdb_dir",
        type=str,
        help="Path to the directory containing original PDB files.",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=None,
        help="Optional total number of lines for tqdm progress bar.",
    )

    args = parser.parse_args()
    run_conversion(args.foldseek_path, args.output_path, args.pdb_dir, total=args.total)
