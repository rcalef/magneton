from pprint import pprint
from typing import Dict, Tuple, Type

import lightning as L

from magneton.config import EmbeddingConfig
from .base_embedder import BaseConfig, BaseDataModule, BaseEmbedder
from .esm_embedder import ESMEmbedder
from .gearnet_embedder import GearNetEmbedder
from .esmc_embedder import ESMCEmbedder, ESMCConfig, ESMCDataModule

class EmbedderFactory:
    _embedders: Dict[str, Tuple[Type[BaseEmbedder], Type[BaseConfig]], Type[BaseDataModule]] = {
        "esm": (ESMEmbedder, None, None),
        "gearnet": (GearNetEmbedder, None, None),
        "esmc": (ESMCEmbedder, ESMCConfig, ESMCDataModule),
    }

    @classmethod
    def register_embedder(cls, name: str, embedder_class: Type[BaseEmbedder]):
        """Register a new embedder type"""
        cls._embedders[name] = embedder_class

    @classmethod
    def create_embedder(cls, config: EmbeddingConfig) -> Tuple[BaseEmbedder, Type[BaseDataModule]]:
        """Create an embedder instance based on config"""
        model_type = config.model
        embedder_class, config_class, data_module = cls._embedders.get(model_type)

        if embedder_class is None:
            raise ValueError(f"Unknown embedder type: {model_type}")

        # Print debug info
        print(f"\n=== Creating {model_type} embedder ===")
        print(f"Config parameters: {pprint(config)}")

        embedder_config = config_class(
            device=config.device,
            **config.model_params,
        )

        return embedder_class(embedder_config), data_module