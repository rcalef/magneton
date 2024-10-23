from dataclasses import dataclass
from pprint import pprint
from typing import List, Tuple


@dataclass
class InterproEntry:
    id: str
    element_type: str
    match_id: str
    element_name: str
    representative: bool
    positions: List[Tuple[int]]

@dataclass
class Protein:
    uniprot_id: str
    kb_id: str
    name: str
    length: int
    parsed_entries: int
    total_entries: int
    entries: List[InterproEntry]

    def print(self):
        pprint(self)