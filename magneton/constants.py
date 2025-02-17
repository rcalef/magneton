from enum import auto, IntEnum, StrEnum

class DataType(StrEnum):
    SEQ = "sequence"
    STRUCT = "structure"
    SUBSTRUCT = "substructure"

class PipelineStage(IntEnum):
    EMBED = 0
    TRAIN = auto()
    VISUALIZE = auto()

stage_names = [
    "embed",
    "train",
    "visualize",
]

# Map stage names to their corresponding enum values
name_to_stage = {name: stage for stage, name in enumerate(stage_names)}