import json
import logging
import requests

from multiprocessing import Pool
from pathlib import Path

from Bio.PDB import PDBParser, PPBuilder
from pysam import FastaFile
from six.moves.urllib.request import urlopen
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def pdb_id_to_uniprot(pdb: str, chain: str) -> str | None:
    try:
        content = urlopen('https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/' + pdb).read()
    except Exception as e:
        logger.debug(f"{pdb} {chain} PDB Not Found (HTTP Error 404). Skipped. {e}")
        return None

    content = json.loads(content.decode('utf-8'))

    # find uniprot id
    for uniprot in content[pdb.lower()]['UniProt'].keys():
        for mapping in content[pdb.lower()]['UniProt'][uniprot]['mappings']:
            if mapping['chain_id'] == chain:
                return uniprot

    logger.debug(f"{pdb} {chain} PDB Found but Chain Not Found. Skipped.")
    return None

def download_one_afdb_file(
    path: Path,
) -> bool:
    base_url = "https://alphafold.ebi.ac.uk/files/"
    url = base_url + path.name

    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            with open(path, "wb") as f:
                f.write(r.content)
        else:
            return False
    except Exception as e:
        logger.debug(f"failed to fetch AFDB file ({path}): {e}")
        return False
    return True

def download_afdb_files(
    file_paths: list[Path],
    num_workers: int,
):
    with Pool(num_workers) as p:
        download_results = list(
            tqdm(
                p.imap_unordered(download_one_afdb_file, file_paths),
                total=len(file_paths)
            )
        )
    success = sum(download_results)
    logger.info(f"succesfully downloaded {success}  / {len(file_paths)} files")

def parse_one_seq_from_pdb(
    path: Path,
) -> str:
    parser = PDBParser(QUIET=True)
    ppb = PPBuilder()
    structure = parser.get_structure("protein", path)

        # AFDB structures all consist of one chain, so just take the first one.
        # This seems to be what folks must do for using these classic splits based
        # on PDB IDs (both ProSST and SaProt use AFDB structures instead of the
        # available PDB structures).
    chains = list(list(structure)[0])
    if len(chains) != 1:
        raise ValueError(f"expected single chain, got {len(chains)}: {path}")
    return "".join([str(pp.get_sequence()) for pp in ppb.build_peptides(chains[0])])

def parse_seqs_from_pdbs(
    fasta_path: Path,
    file_paths: list[Path],
    uniprot_ids: list[str],
    num_workers: int,
) -> dict[str, str]:
    """
    Extract sequences from PDBs; cache to FASTA for fast re-loads.
    Returns dict: UniProt -> sequence
    """
    if fasta_path.exists():
        logger.info(f"FASTA cache found at {fasta_path}")
        fa = FastaFile(str(fasta_path))
        return {uid: fa.fetch(uid) for uid in uniprot_ids}

    logger.info("No FASTA cache found, parsing sequences from PDB files")
    with Pool(num_workers) as p:
        sequences = list(tqdm(p.imap(parse_one_seq_from_pdb, file_paths), total=len(file_paths), desc="Parsing PDBs"))

    ret: dict[str, str] = {}
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fasta_path, "w") as fh:
        for uid, seq in zip(uniprot_ids, sequences):
            fh.write(f">{uid}\n{seq}\n")
            ret[uid] = seq
    return ret
