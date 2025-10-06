from dataclasses import dataclass, replace
from pathlib import Path
from typing import Generator

import torch
from torch.utils.data import Dataset
from pysam import FastaFile

from magneton.config import DataConfig
from magneton.core_types import DataType, Protein

from .protein_dataset import get_protein_dataset
from .substructure_parsers import (
    get_substructure_parser,
    LabeledSubstructure,
)

@dataclass(kw_only=True)
class Batch:
    """Collated batch of dataset entries.

    - protein_ids (List[str]): UniProt IDs for all proteins in the batch.
    - lengths (List[int]): Length in AAs for each protein.
    - seqs (List[str] | None): AA sequences for each protein.
    - substructures (List[List[LabeledSubstructure]] | None): For each protein,
      a list of annotated substructures.
    - structure_list (List[str] | None): Paths to structure (.pdb) files for each protein.
    - labels (torch.Tensor | None): Labels for supervised tasks.
    """
    protein_ids: list[str]
    lengths: list[int]
    seqs: list[str] | None = None
    # First element is ranges, second element is labels
    substructures: list[list[LabeledSubstructure]] | None = None
    structure_list: list[str] | None = None
    labels: torch.Tensor | None = None

    def to(self, device: str):
        if self.labels is not None:
            self.labels = self.labels.to(device)
        if self.substructures is not None:
            for i in range(len(self.substructures)):
                for j in range(len(self.substructures[i])):
                    self.substructures[i][j] = self.substructures[i][j].to(device)
        return self

    def total_length(self) -> int:
        return sum(map(len, self.substructures))

@dataclass(kw_only=True)
class DataElement:
    """Single dataset entry.

    - protein_id (str): UniProt ID for this protein.
    - length (int): Length in AAs.
    - seq (str | None): AA sequence of protein.
    - substructures (List[LabeledSubstructure] | None): Annotated substructures.
    - structure_path (str | None): Path to structure (.pdb) file.
    - labels: (torch.Tensor | None): Labels for supervised tasks.
    """
    protein_id: str
    length: int
    seq: str | None = None
    # First element is ranges, second element is labels
    substructures: list[LabeledSubstructure] | None = None
    structure_path: str | None = None
    # For any downstream supervised tasks
    labels: torch.Tensor | None = None


class CoreDataset(Dataset):
    """Dataset that yields proteins and associated sequence, structure, and substructures.

    This is the main Dataset object for the Magneton datasets. The individual elements are
    `DataElement` objects as defined above, converted from the serialized `core_types.Protein`
    objects. The types of data contained in each `DataElement` are configured via the
    `want_datatypes` argument.

    Args:
        - data_config (DataConfig): Config specifying data file locations.
        - want_datatypes (list[DataType]): List of desired datatypes for returned elements.
        - load_fasta_in_mem (bool): Whether to load full FASTA file of sequences into memory
            or to read on the fly using pysam.
    """
    def __init__(
        self,
        data_config: DataConfig,
        want_datatypes: list[DataType],
        load_fasta_in_mem: bool = True,
    ):
        super().__init__()

        self.datatypes = set(want_datatypes)
        if DataType.SEQ in self.datatypes:
            assert data_config.fasta_path is not None, "Fasta path is required for sequence data"
            if load_fasta_in_mem:
                fa = FastaFile(data_config.fasta_path)
                self.fasta = {k: fa.fetch(k) for k in fa.references}
            else:
                self.fasta = FastaFile(data_config.fasta_path)
        if DataType.SUBSTRUCT in self.datatypes:
            assert data_config.labels_path is not None, "Labels path is required for substructure data"
            if not data_config.collapse_labels and len(data_config.substruct_types) == 1:
                print(
                    "Warning: collapse_labels is set to False, but only one InterPro type is provided.\n"
                    "Forcing collapse_labels to True for simplicity."
                )
                data_config.collapse_labels = True
            self.substruct_parser = get_substructure_parser(data_config)

        if DataType.STRUCT in self.datatypes:
            assert data_config.struct_template is not None, "Structure path is required for structure data"
            self.struct_template = data_config.struct_template

        self.dataset = get_protein_dataset(
            input_path=data_config.data_dir,
            compression=data_config.compression,
            prefix=data_config.prefix,
            want_subtype_parser=self.substruct_parser if DataType.SUBSTRUCT in self.datatypes else None,
            # TODO: make large protein datasets random access
            in_memory=True,
        )

    def _prot_to_elem(self, prot: Protein) -> DataElement:
        ret = DataElement(protein_id=prot.uniprot_id, length=prot.length)
        if DataType.SEQ in self.datatypes:
            ret.seq = self.fasta[prot.kb_id]
        if DataType.SUBSTRUCT in self.datatypes:
            ret.substructures = self.substruct_parser.parse(prot)
        if DataType.STRUCT in self.datatypes:
            # Extract uniprot ID from protein ID (removing any additional identifiers)
            uniprot_id = prot.uniprot_id.split('|')[0] if '|' in prot.uniprot_id else prot.uniprot_id
            ret.structure_path = self.struct_template % uniprot_id
        return ret

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Generator[DataElement, None, None]:
        for prot in self.dataset:
            yield self._prot_to_elem(prot)

    def __getitem__(self, index):
        return self._prot_to_elem(self.dataset[index])

def get_core_dataset(
    data_config: DataConfig,
    want_datatypes: list[DataType],
    split: str = "train",
    load_fasta_in_mem: bool = True,
) -> Dataset:
    """Convenience function for getting dataset for a given split."""
    if split != "all":
        split_dir = Path(data_config.data_dir) / f"{split}_sharded"
        prefix = f"swissprot.with_ss.{split}"
        data_config = replace(
            data_config,
            data_dir=split_dir,
            prefix=prefix,
        )

    prot_dataset = CoreDataset(
        data_config=data_config,
        want_datatypes=want_datatypes,
        load_fasta_in_mem=load_fasta_in_mem,
    )
    return prot_dataset