from .deepfri_dataset import DeepFriModule
from .flip_dataset import FlipModule
from .peer_dataset import PeerDataModule, PEER_TASK_TO_CONFIGS
from .protein_workshop_dataset import WorkshopDataModule, TASK_TO_CONFIGS
from .saprot_dataset import DeepLocModule, ThermostabilityModule
from .task_types import (
    EVAL_TASK,
    TASK_GRANULARITY,
    TASK_TO_TYPE,
    TASK_TYPES
)