from typing import Dict, Type
from .base_embedder import BaseEmbedder
from .esm_embedder import ESMEmbedder
from .gearnet_embedder import GearNetEmbedder
from .esmc_embedder import ESMCEmbedder, ESMCConfig
from omegaconf import DictConfig

class EmbedderFactory:
    _embedders: Dict[str, Type[BaseEmbedder]] = {
        "esm": ESMEmbedder,
        "gearnet": GearNetEmbedder,
        "esmc": ESMCEmbedder
    }
    
    @classmethod
    def register_embedder(cls, name: str, embedder_class: Type[BaseEmbedder]):
        """Register a new embedder type"""
        cls._embedders[name] = embedder_class
        
    @classmethod
    def create_embedder(cls, config: DictConfig) -> BaseEmbedder:
        """Create an embedder instance based on config"""
        model_type = config.get('model_type', 'esmc')  # default to esmc
        embedder_class = cls._embedders.get(model_type)
        
        if embedder_class is None:
            raise ValueError(f"Unknown embedder type: {model_type}")
            
        # Print debug info
        print(f"\n=== Creating {model_type} embedder ===")
        print(f"Config parameters: {dict(config)}")
        
        # Convert generic config to specific embedder config
        if model_type == "esmc":
            specific_config = ESMCConfig(
                weights_path=config.get('weights_path', 'path/to/weights'),
                use_flash_attn=config.get('use_flash_attn', False),
                rep_layer=config.get('rep_layer', 35),
                max_seq_length=config.get('max_seq_length', 2048),
                batch_size=config.get('batch_size', 32),
                device=config.get('device', 'cuda')
            )
        else:
            specific_config = config
            
        return embedder_class(specific_config)