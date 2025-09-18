from pprint import pprint
from typing import Dict, Tuple, Type

from magneton.config import EmbeddingConfig
from .base_embedder import BaseConfig, BaseEmbedder
from .esm2_embedder import ESM2Embedder, ESM2Config
from .esmc_embedder import ESMCEmbedder, ESMCConfig
from .prosst_embedder import ProSSTEmbedder, ProSSTConfig
from .saprot_embedder import SaProtEmbedder, SaProtConfig


class EmbedderFactory:
    _embedders: Dict[str, Tuple[Type[BaseEmbedder], Type[BaseConfig]]] = {
        "esm2": (ESM2Embedder, ESM2Config),
        "esmc": (ESMCEmbedder, ESMCConfig),
        "prosst": (ProSSTEmbedder, ProSSTConfig),
        "saprot": (SaProtEmbedder, SaProtConfig)
    }

    @classmethod
    def register_embedder(
        cls,
        name: str,
        embedder_class: Type[BaseEmbedder],
        config_class: Type[BaseConfig],
    ):
        """Register a new embedder type"""
        cls._embedders[name] = (embedder_class, config_class)

    @classmethod
    def fetch_embedder_classes(cls, name: str) -> Tuple[Type[BaseEmbedder], Type[BaseConfig]]:
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
        if model_type not in cls._embedders:
            raise ValueError(f"Unknown embedder type: {model_type}")

        embedder_class, config_class = cls._embedders.get(model_type)

        # Print debug info
        print(f"\n=== Creating {model_type} embedder ===")
        print(f"Config parameters: {pprint(config)}")

        embedder_config = config_class(
            **config.model_params,
        )

        return embedder_class(embedder_config, frozen=frozen)