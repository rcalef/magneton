from dataclasses import dataclass, field
from enum import auto, IntEnum, StrEnum, unique
from pprint import pprint
from typing import List, Tuple


@unique
class DsspType(IntEnum):
    H = 0
    B = auto()
    E = auto()
    G = auto()
    I = auto()
    P = auto()
    T = auto()
    S = auto()
    # In place of ' ' (space) for OTHER
    X = auto()


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
    dssp_type: DsspType
    # Note that positions are 1-indexed, as output by dssp.
    # Coordinates are half-open, i.e. [start, end)
    start: int
    end: int

    def print(self):
        print(f"{DSSP_TO_NAME[self.dssp_type.value]}: {self.start} - {self.end}")


class InterProType(StrEnum):
    FAMILY = "Family"
    DOMAIN = "Domain"
    HOMO_FAMILY = "Homologous_superfamily"
    CONS_SITE = "Conserved_site"
    ACT_SITE = "Active_site"
    BIND_SITE = "Binding_site"
    PTM = "PTM"

# InterPro types that use the `representative` field
INTERPRO_REP_TYPES = [
    InterProType.FAMILY,
    InterProType.DOMAIN,
]

@dataclass
class InterproEntry:
    id: str
    element_type: InterProType
    match_id: str
    element_name: str
    representative: bool
    # Note that positions are 1-indexed, i.e. exactly as given in InterPro.
    positions: List[Tuple[int]]

    def print(self):
        pprint(self)


@dataclass
class Protein:
    uniprot_id: str
    kb_id: str
    name: str
    length: int
    parsed_entries: int
    total_entries: int
    entries: List[InterproEntry]
    secondary_structs: List[SecondaryStructure] = field(default_factory=list)

    def print(self):
        pprint(self)

    def __setstate__(self, state):
        return self.__init__(**state)


class DataType(StrEnum):
    SEQ = "sequence"
    STRUCT = "structure"
    SUBSTRUCT = "substructure"


class PipelineStage(IntEnum):
    EMBED = 0
    TRAIN = auto()
    EVALUATE = auto()


stage_names = [
    "embed",
    "train",
    "eval",
]

# Map stage names to their corresponding enum values
name_to_stage = {name: PipelineStage(stage) for stage, name in enumerate(stage_names)}
