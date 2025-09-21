from enum import StrEnum


class TASK_GRANULARITY(StrEnum):
    PROTEIN_CLASSIFICATION = "protein_classification"
    RESIDUE_CLASSIFICATION = "residue_classification"
    CONTACT_PREDICTION = "contact_prediction"
    PPI_PREDICTION = "ppi_prediction"


class EVAL_TASK(StrEnum):
    MULTILABEL = "multilabel"
    MULTICLASS = "multiclass"
    BINARY = "binary"
    REGRESSION = "regression"


TASK_TYPES = {
    EVAL_TASK.MULTILABEL: [
        "EC",
        "GO:BP",
        "GO:CC",
        "GO:MF",
    ],
    EVAL_TASK.MULTICLASS: ["fold", "saprot_subloc", "FLIP_bind"],
    EVAL_TASK.BINARY: [
        "biolip_binding",
        "biolip_catalytic",
        "binary_localization",
        "contact_prediction",
        "human_ppi",
        "saprot_binloc",
        "solubility",
    ],
    EVAL_TASK.REGRESSION: [
        "fluorescence",
        "stability",
        "beta_lactamase",
        "aav",
        "gb1",
        "thermostability",
        "subcellular_localization",
        "saprot_thermostability",
    ],
}
TASK_TO_TYPE = {
    task: task_type for task_type, tasks in TASK_TYPES.items() for task in tasks
}
