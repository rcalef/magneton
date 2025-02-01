from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict, Any
import torch
from ..config.base_config import EmbeddingConfig

class BaseEmbedder(ABC):
    """Base class for protein embedders"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.device)
        
    @abstractmethod
    def embed_one_prot(self, protein_data: Any, max_len: Optional[int] = None) -> torch.Tensor:
        """Embed a single protein"""
        pass
        
    @abstractmethod
    def embed_sequences(self, sequences: List[str]) -> List[torch.Tensor]:
        """Embed multiple sequences"""
        pass
        
    @abstractmethod
    def process_protein(self, protein_data: Any) -> torch.Tensor:
        """Process a single protein through the model"""
        pass
        
    @classmethod
    def get_required_input_type(cls) -> str:
        """Return the required input type for this embedder"""
        pass 