from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Set
import torch

from magneton.constants import DataType

# from ..config.base_config import EmbeddingConfig


@dataclass
class BaseConfig:
    device: str = field(kw_only=True, default="cpu")


class BaseEmbedder(ABC):
    """Base class for protein embedders"""

    def __init__(self, config: BaseConfig):
        self.config = config
        self.device = torch.device(config.device)

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
    ) -> List[torch.Tensor]:
        """Embed multiple sequences"""
        pass

    # @abstractmethod
    def process_protein(self, protein_data: Any) -> torch.Tensor:
        """Process a single protein through the model"""
        pass

    @classmethod
    def get_required_input_type(cls) -> Set[DataType]:
        """Return the required input type for this embedder"""
        pass

    @classmethod
    def model_name(cls) -> str:
        """Return human-readable name for embedder"""
        pass
