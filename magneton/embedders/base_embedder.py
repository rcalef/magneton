from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Set

import torch
import torch.nn as nn
import lightning as L

from magneton.config import DataConfig, TrainingConfig
from magneton.types import DataType

# from ..config.base_config import EmbeddingConfig


class BaseDataModule(L.LightningDataModule, ABC):

    def __init__(
        self,
        data_config: DataConfig,
        train_config: TrainingConfig,
    ):
        super().__init__()
        self.data_config = data_config
        self.train_config = train_config

@dataclass
class BaseConfig:
    device: str = field(kw_only=True, default="cpu")

class BaseEmbedder(nn.Module, ABC):
    """Base class for protein embedders"""

    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config
        # self.device = torch.device(config.device)

    @abstractmethod
    def embed_single_protein(
        self,
        protein_data: Any,
    ) -> torch.Tensor:
        """Embed a single protein"""
        pass

    @abstractmethod
    def embed_batch(
        self,
        batch: List[Any],
    ) -> torch.Tensor:
        """Embed multiple sequences"""
        pass

    # @abstractmethod
    def process_protein(self, protein_data: Any) -> torch.Tensor:
        """Process a single protein through the model"""
        pass

    @abstractmethod
    def get_embed_dim(cls) -> int:
        pass

    @classmethod
    def get_required_input_type(cls) -> Set[DataType]:
        """Return the required input type for this embedder"""
        pass

    @classmethod
    def model_name(cls) -> str:
        """Return human-readable name for embedder"""
        pass
