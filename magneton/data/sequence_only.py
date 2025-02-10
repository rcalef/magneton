from typing import Generator, List, Tuple

from pysam import FastaFile
from torch.utils.data import IterableDataset

from magneton.data.dataset import get_protein_dataset
from magneton.types import Protein

class SequenceOnlyDataset(IterableDataset):
    def __init__(
        self,
        input_path: str,
        fasta_path: str,
        compression: str = "bz2",
        prefix: str = "sharded_proteins",
    ):
        super().__init__()
        self.dataset = get_protein_dataset(
            input_path=input_path,
            compression=compression,
            prefix=prefix,
        )
        self.fasta = FastaFile(fasta_path)

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Generator[Tuple[str, Protein], None, None]:
        for prot in self.dataset:
            yield (self.fasta[prot.kb_id], prot)


def collate_sequence_datasets(
    entries: List[Tuple[str, Protein]],
) -> Tuple[List[str], List[Protein]]:
    # Nothing to do here except override default collator
    return entries
