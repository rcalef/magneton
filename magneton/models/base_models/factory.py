from magneton.config import BaseModelConfig

from .esm2 import ESM2BaseModel, ESM2Config
from .esmc import ESMCBaseModel, ESMCConfig
from .interface import BaseConfig, BaseModel
from .prosst import ProSSTConfig, ProSSTEmbedder
from .saprot import SaProtBaseModel, SaProtConfig
from .s_plm_benchmark_only import SPLMBaseModel, SPLMConfig


class BaseModelFactory:
    _base_models: dict[str, tuple[type[BaseModel], type[BaseConfig]]] = {
        "esm2": (ESM2BaseModel, ESM2Config),
        "esmc": (ESMCBaseModel, ESMCConfig),
        "prosst": (ProSSTEmbedder, ProSSTConfig),
        "saprot": (SaProtBaseModel, SaProtConfig),
        "s-plm": (SPLMBaseModel, SPLMConfig),
    }

    @classmethod
    def create_base_model(
        cls,
        config: BaseModelConfig,
        frozen: bool = True,
    ) -> tuple[BaseModel]:
        """Create an embedder instance based on config"""
        model_type = config.model
        if model_type not in cls._base_models:
            raise ValueError(f"Unknown base model type: {model_type}")

        base_model_class, config_class = cls._base_models.get(model_type)
        base_model_config = config_class(
            **config.model_params,
        )

        return base_model_class(base_model_config, frozen=frozen)
