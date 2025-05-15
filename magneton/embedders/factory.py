from pprint import pprint
from typing import Dict, Tuple, Type

import lightning as L

from magneton.config import EmbeddingConfig
from .base_embedder import BaseConfig, BaseDataModule, BaseEmbedder
from .esm_embedder import ESMEmbedder
from .gearnet_embedder import GearNetEmbedder, GearNetConfig, GearNetDataModule
from .esmc_embedder import ESMCEmbedder, ESMCConfig, ESMCDataModule

class EmbedderFactory:
    _embedders: Dict[str, Tuple[Type[BaseEmbedder], Type[BaseConfig], Type[BaseDataModule]]] = {
        "esm": (ESMEmbedder, None, None),
        "gearnet": (GearNetEmbedder, GearNetConfig, GearNetDataModule),
        "esmc": (ESMCEmbedder, ESMCConfig, ESMCDataModule),
    }

    @classmethod
    def register_embedder(
        cls,
        name: str,
        embedder_class: Type[BaseEmbedder],
        config_class: Type[BaseConfig],
        data_module: Type[BaseDataModule],
    ):
        """Register a new embedder type"""
        cls._embedders[name] = (embedder_class, config_class, data_module)

    @classmethod
    def fetch_embedder_classes(cls, name: str) -> Tuple[Type[BaseEmbedder], Type[BaseConfig], Type[BaseDataModule]]:
        """Register a new embedder type"""
        if name not in cls._embedders:
            raise ValueError(f"Unknown embedder type: {name}")
        return cls._embedders[name]

    @classmethod
    def create_embedder(
        cls,
        config: EmbeddingConfig,
        frozen: bool = True,
    ) -> Tuple[BaseEmbedder]:
        """Create an embedder instance based on config"""
        model_type = config.model
        embedder_class, config_class, _ = cls._embedders.get(model_type)

        if embedder_class is None:
            raise ValueError(f"Unknown embedder type: {model_type}")

        # Print debug info
        print(f"\n=== Creating {model_type} embedder ===")
        print(f"Config parameters: {pprint(config)}")

        embedder_config = config_class(
            **config.model_params,
        )

        return embedder_class(embedder_config, frozen=frozen)