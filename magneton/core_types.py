from dataclasses import dataclass, field
from enum import IntEnum, StrEnum, auto, unique
from pprint import pprint


@unique
class DsspType(IntEnum):
    H = 0
    B = auto()
    E = auto()
    G = auto()
    I = auto()  # noqa: E741
    P = auto()
    T = auto()
    S = auto()
    # In place of ' ' (space) for OTHER
    X = auto()


# Conversions between single letter DSSP codes
# and human-readable or MMCIF structure names.
DSSP_TO_NAME = [
    "Alphahelix",
    "Betabridge",
    "Strand",
    "Helix_3",
    "Helix_5",
    "Helix_PPII",
    "Turn",
    "Bend",
    "Loop",
]
NAME_TO_DSSP = {name: i for i, name in enumerate(DSSP_TO_NAME)}

DSSP_TO_MMCIF = [
    "HELX_RH_AL_P",
    "STRN",
    "STRN",
    "HELX_RH_3T_P",
    "HELX_RH_PI_P",
    "HELX_LH_PP_P",
    "TURN_TY1_P",
    "BEND",
    "OTHER",
]

MMCIF_TO_DSSP = {mmcif: i for i, mmcif in enumerate(DSSP_TO_MMCIF)}


@dataclass
class SecondaryStructure:
    """Representation of a single DSSP secondary structure"""

    dssp_type: DsspType
    # Note that positions are 1-indexed, as output by dssp.
    # Coordinates are half-open, i.e. [start, end)
    start: int
    end: int

    def print(self):
        print(f"{DSSP_TO_NAME[self.dssp_type.value]}: {self.start} - {self.end}")

    def toJSON(self) -> dict:
        return {
            "dssp_type": self.dssp_type.value,
            "start": self.start,
            "end": self.end,
        }

    @classmethod
    def fromJSON(cls, data: dict) -> "SecondaryStructure":
        return cls(
            dssp_type=DsspType(data["dssp_type"]),
            start=data["start"],
            end=data["end"],
        )


class SubstructType(StrEnum):
    """Enum of available substructure types"""

    FAMILY = "Family"
    DOMAIN = "Domain"
    HOMO_FAMILY = "Homologous_superfamily"
    CONS_SITE = "Conserved_site"
    ACT_SITE = "Active_site"
    BIND_SITE = "Binding_site"
    PTM = "PTM"
    SS = "Secondary_struct"
    REPEAT = "Repeat"


# InterPro types that use the `representative` field
INTERPRO_REP_TYPES = [
    SubstructType.FAMILY,
    SubstructType.DOMAIN,
    SubstructType.REPEAT,
]


@dataclass
class InterproEntry:
    """A single InterPro entry.

    - id (str): InterPro ID for this entry's element.
    - element_type (InterProType): Type of InterPro element.
    - match_id (str): ID of the specific match.
    - element_name (str): Human-readable name of the InterPro element.
    - representative (bool): Whether or not this is the representative entry of this element for this protein.
    - positions (List[Tuple[int]]): List of [start, end) positions for this entry, 1-indexed as in InterPro.
    """

    id: str
    element_type: SubstructType
    match_id: str
    element_name: str
    representative: bool
    # Note that positions are 1-indexed, i.e. exactly as given in InterPro.
    positions: list[tuple[int, int]]

    def print(self):
        pprint(self)

    def toJSON(self) -> dict:
        return {
            "id": self.id,
            "element_type": self.element_type,
            "match_id": self.match_id,
            "element_name": self.element_name,
            "representative": self.representative,
            # tuples -> lists for JSON
            "positions": [list(p) for p in self.positions],
        }

    @classmethod
    def fromJSON(cls, data: dict) -> "InterproEntry":
        return cls(
            id=data["id"],
            element_type=SubstructType(data["element_type"]),
            match_id=data["match_id"],
            element_name=data["element_name"],
            representative=data["representative"],
            # lists -> tuples
            positions=[tuple(p) for p in data["positions"]],
        )


@dataclass
class Protein:
    """A single parsed protein.

    - uniprot_id (str): UniProt ID of the protein.
    - kb_id (str): UniProtKB ID of the protein (for fasta indexing)
    - name (str): Human-readable name of protein, if any
    - length (int): Length in AAs.
    - parsed_entries (int): Number of parsed substructure entries.
    - total_entries (int): Total number of substructure entries, including filtered entries.
    - entries (List[InterproEntry]): List of InterPro substructure entries
    - secondary_structs (List[SecondaryStructure]): List of secondary structures.
    """

    uniprot_id: str
    kb_id: str
    name: str
    length: int
    parsed_entries: int
    total_entries: int
    entries: list[InterproEntry]
    secondary_structs: list[SecondaryStructure] = field(default_factory=list)

    def print(self):
        pprint(self)

    def __setstate__(self, state):
        return self.__init__(**state)

    def toJSON(self) -> dict:
        return {
            "uniprot_id": self.uniprot_id,
            "kb_id": self.kb_id,
            "name": self.name,
            "length": self.length,
            "parsed_entries": self.parsed_entries,
            "total_entries": self.total_entries,
            "entries": [e.toJSON() for e in self.entries],
            "secondary_structs": [s.toJSON() for s in self.secondary_structs],
        }

    @classmethod
    def fromJSON(cls, data: dict) -> "Protein":
        return cls(
            uniprot_id=data["uniprot_id"],
            kb_id=data["kb_id"],
            name=data["name"],
            length=data["length"],
            parsed_entries=data["parsed_entries"],
            total_entries=data["total_entries"],
            entries=[InterproEntry.fromJSON(e) for e in data["entries"]],
            secondary_structs=[
                SecondaryStructure.fromJSON(s) for s in data["secondary_structs"]
            ],
        )


class DataType(StrEnum):
    """Enum used to define the data types a given model requires."""

    SEQ = "sequence"
    STRUCT = "structure"
    SUBSTRUCT = "substructure"
