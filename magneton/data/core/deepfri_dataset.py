import json
import logging
import requests

from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Literal

import pandas as pd
import torch
from Bio.PDB import PDBParser, PPBuilder
from torch.utils.data import (
    Dataset,
)
from pysam import FastaFile
from six.moves.urllib.request import urlopen
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from .core_dataset import DataElement

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupervisedDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        num_classes: int,
    ):
        self.df = df
        self.num_classes = num_classes

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        # Sum first dim because of multilabel classification, i.e.
        # if this item has N labels (e.g. N GO terms), this does:
        # (N X num_classes) -> (num_classes)
        one_hot_labels = torch.nn.functional.one_hot(
            row.labels,
            num_classes=self.num_classes,
        ).sum(dim=0)
        return DataElement(
            protein_id=row.protein_id,
            length=len(row.seq),
            labels=one_hot_labels,
            seq=row.seq,
            structure_path=str(row.structure_path),
        )


def _map_one_pdb_id(pdb: str, chain: str) -> str | None:
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

def _download_one_file(
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

def _parse_one_pdb_seq(
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

LABEL_LINE: Dict[str, int] = {
    "MF": 1,
    "BP": 5,
    "CC": 9,
    "EC": 1,
}

NUM_SUPERVISED_CLASSES: Dict[str, int] = {
    "BP": 1943,
    "CC": 320,
    "MF": 489,
    "EC": 538,
}


class DeepFriModule:
    def __init__(
        self,
        task: str,
        data_dir: str,
        struct_template: str,
        num_workers: int = 16,
    ):
        self.task = task
        self.data_dir = Path(data_dir)
        self.num_workers = num_workers
        self.struct_template = struct_template

    def num_classes(self) -> int:
        return NUM_SUPERVISED_CLASSES[self.task]

    def get_dataset(self, split: str) -> Dataset:
        dataset_df = self.get_deepfri_df(split)
        num_classes = NUM_SUPERVISED_CLASSES[self.task]
        return SupervisedDataset(df=dataset_df, num_classes=num_classes)

    def get_deepfri_df(
        self,
        split: Literal[
            "train",
            "val",
            "test"
        ],
    ) -> pd.DataFrame:
        """Data module encompassing evaluation sets from DeepFRI (GO terms and EC numbers)

        Structure largely adapted from ProteinWorkshop.
        """
        if self.task == "EC":
            data_dir = self.data_dir / "EnzymeCommission"

            label_fname = data_dir / "nrPDB-EC_2020.04_annot.tsv"
            train_fname = data_dir / "nrPDB-EC_2020.04_train.txt"
            val_fname = data_dir / "nrPDB-EC_2020.04_valid.txt"
            test_fname = data_dir / "nrPDB-EC_2020.04_test.txt"

            n_header_rows = 3
            col_names = ["PDB", "EC"]
        else:
            data_dir = self.data_dir / "GeneOntology"

            label_fname = data_dir / "nrPDB-GO_annot.tsv"
            train_fname = data_dir / "nrPDB-GO_train.txt"
            val_fname = data_dir / "nrPDB-GO_valid.txt"
            test_fname = data_dir / "nrPDB-GO_test.txt"

            n_header_rows = 13
            col_names = ["PDB", "MF", "BP", "CC"]


        # Parse labels
        label_line = LABEL_LINE[self.task]
        with open(label_fname, "r") as f:
            all_labels = f.readlines()[label_line].strip("\n").split("\t")
        labeller = LabelEncoder().fit(all_labels)
        num_classes = len(labeller.classes_)
        assert num_classes == NUM_SUPERVISED_CLASSES[self.task]

        # Load labels for full dataset
        df = (
            pd.read_table(
                label_fname,
                skiprows=n_header_rows,
                names=col_names,
            )
            .loc[lambda x: ~x[self.task].isna()]
            .assign(
                labels=lambda x: x[self.task].str.split(",").map(labeller.transform),
                labels_tens=lambda x: x.labels.map(torch.tensor),
            )
            .drop(columns=["labels"])
            .rename(columns={"PDB": "pdb_id", "labels_tens": "labels"})
        )
        dataset = df[["pdb_id", self.task, "labels"]]

        # Add in PDB IDs
        pdb_id_map_path = data_dir / f"{self.task}.pdb_to_uniprot.tsv"
        pdb_id_map = self.map_pdb_ids(dataset.pdb_id, pdb_id_map_path)
        dataset = dataset.merge(pdb_id_map, on="pdb_id")

        # Add in structure paths
        dataset = dataset.assign(
            structure_path=dataset.uniprot_id.apply(
                lambda y: Path(self.struct_template % y) if y is not None else None
            )
        )

        # Download any missing structures
        file_paths = dataset.structure_path.dropna().to_list()
        missing_files = [path for path in file_paths if not path.exists()]
        logger.info(f"downloading {len(missing_files)} / {len(file_paths)} missing files")
        self.download_struct_files(missing_files)

        # Remove entries with no structures
        dataset = dataset.loc[
            dataset.structure_path.notna() & dataset.structure_path.apply(lambda p: p.exists())
        ]

        # Extract sequences from PDB files to make sure they match the downloaded
        # structure
        sequences = self.extract_sequence_from_pdbs(dataset.structure_path.to_list())
        dataset = dataset.assign(seq=sequences)

        if split == "train":
            data = pd.read_csv(train_fname, sep="\t", header=None)[0]
        elif split == "val":
            data = pd.read_csv(val_fname, sep="\t", header=None)[0]
        elif split == "test":
            data = pd.read_csv(test_fname, sep="\t", header=None)[0]
        else:
            raise ValueError(f"Unknown split: {split}")

        want_subset = (
            dataset
            .loc[lambda x: x.pdb_id.isin(data)]
            .rename(columns={"pdb_id": "protein_id"})
        )
        return want_subset

    def map_pdb_ids(
        self,
        pdb_ids: pd.Series,
        path: Path,
    ) -> pd.DataFrame:
        if path.exists():
            logger.info(f"found PDB-UniProt mapping at: {path}")
            return pd.read_table(path)

        ids_and_chains = pdb_ids.str.split("-").to_list()
        logger.info(f"mapping PDB IDs for {len(ids_and_chains)} proteins")
        with Pool(self.num_workers) as p:
            uniprot_ids = p.starmap(_map_one_pdb_id, tqdm(ids_and_chains))

        id_map = pd.DataFrame({
            "pdb_id": pdb_ids,
            "uniprot_id": uniprot_ids,
        })
        logger.info(f"mapped PDB IDs for {id_map.uniprot_id.notna().sum()} / {len(id_map)}")
        id_map.to_csv(path, sep="\t", index=False)
        return id_map

    def download_struct_files(
        self,
        file_paths: list[Path],
    ):
        with Pool(self.num_workers) as p:
            download_results = list(
                tqdm(
                    p.imap_unordered(_download_one_file, file_paths),
                    total=len(file_paths)
                )
            )
        success = sum(download_results)
        logger.info(f"succesfully downloaded {success}  / {len(file_paths)} files")

    def extract_sequence_from_pdbs(
        self,
        file_paths: list[Path],
    ) -> list[str]:
        logger.info("Parsing sequences from PDB files")
        with Pool(self.num_workers) as p:
            sequences = list(tqdm(p.imap(_parse_one_pdb_seq, file_paths), total=len(file_paths)))
        return sequences