import os
from dataclasses import dataclass
from typing import Generator, List

from pysam import FastaFile
import torch
from torch.utils.data import Dataset

from magneton.config import DataConfig
from magneton.data.protein_dataset import get_protein_dataset
from magneton.data.substructure import LabeledSubstructure, SubstructureParser
from magneton.types import DataType, InterProType, Protein


@dataclass
class Batch:
    protein_ids: List[str]
    seqs: List[str] | None
    # First element is ranges, second element is labels
    substructures: List[List[LabeledSubstructure]] | None
    structure_list: List[str] | None = None

@dataclass
class BatchElement:
    protein_id: str
    seq: str | None = None
    # First element is ranges, second element is labels
    substructures: List[LabeledSubstructure] | None = None
    structure_path: str | None = None


class MetaDataset(Dataset):
    def __init__(
        self,
        data_config: DataConfig,
        want_datatypes: List[DataType],
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
            self.substruct_parser = SubstructureParser(
                want_types=data_config.interpro_types,
                labels_dir=data_config.labels_path,
            )
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

    def _prot_to_elem(self, prot: Protein) -> BatchElement:
        ret = BatchElement(protein_id=prot.uniprot_id)
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

    def __iter__(self) -> Generator[BatchElement, None, None]:
        for prot in self.dataset:
            yield self._prot_to_elem(prot)

    def __getitem__(self, index):
        return self._prot_to_elem(self.dataset[index])

def collate_meta_datasets(
    entries: List[BatchElement],
    filter_empty_substruct=True,
) -> Batch:
    """
    Collate the entries into a batch.
    """
    if filter_empty_substruct and entries[0].substructures is not None:
        entries = [x for x in entries if len(x.substructures) > 0]
    batch = Batch(
        protein_ids=[e.protein_id for e in entries],
        seqs=[e.seq for e in entries] if entries[0].seq is not None else None,
        substructures=[e.substructures for e in entries] if entries[0].substructures is not None else None,
        structure_list=[e.structure_path for e in entries] if entries[0].structure_path is not None else None,
    )
    return batch
