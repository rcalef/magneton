from typing import Dict, Type
from .base_embedder import BaseEmbedder
from .esm_embedder import ESMEmbedder
from .gearnet_embedder import GearNetEmbedder
from ..config.base_config import EmbeddingConfig

# TODO
# Replace with using embedder directly in pipeline like so:
# from magneton.embedders import ESMEmbedder, GearNetEmbedder
# from magneton.config.base_config import ESMConfig, GearNetConfig

# # ESM embedder
# esm_config = ESMConfig(
#     model_type="esm",
#     model_name="esm3_sm_open_v1",
#     max_seq_length=1024
# )
# esm_embedder = ESMEmbedder(esm_config)

# # GearNet embedder
# gearnet_config = GearNetConfig(
#     model_type="gearnet",
#     max_seq_length=350,
#     hidden_dims=[512, 512, 512]
# )
# gearnet_embedder = GearNetEmbedder(gearnet_config)

class EmbedderFactory:
    _embedders: Dict[str, Type[BaseEmbedder]] = {
        "esm": ESMEmbedder,
        "gearnet": GearNetEmbedder
    }
    
    @classmethod
    def register_embedder(cls, name: str, embedder_class: Type[BaseEmbedder]):
        """Register a new embedder type"""
        cls._embedders[name] = embedder_class
        
    @classmethod
    def create_embedder(cls, config: EmbeddingConfig) -> BaseEmbedder:
        """Create an embedder instance based on config"""
        embedder_class = cls._embedders.get(config.model_type)
        if embedder_class is None:
            raise ValueError(f"Unknown embedder type: {config.model_type}")
        return embedder_class(config)