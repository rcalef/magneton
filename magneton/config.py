from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

@dataclass
class EmbeddingConfig:
    _target_: str = "magneton.config.EmbeddingConfig"
    model: str = MISSING
    batch_size: int = 32
    device: str = "cuda"
    model_params: Optional[Dict[str, Any]] = field(default_factory=dict)

@dataclass
class DataConfig:
    _target_: str = "magneton.config.DataConfig"
    data_dir: str = MISSING
    fasta_path: Optional[str] = None
    compression: str = "bz2"
    prefix: str = "sharded_proteins"

@dataclass
class ModelConfig:
    _target_: str = "magneton.config.ModelConfig"
    model_type: str = MISSING

@dataclass
class TrainingConfig:
    _target_: str = "magneton.config.TrainingConfig"
    max_epochs: int = 100

@dataclass
class PipelineConfig:
    _target_: str = "magneton.config.PipelineConfig"
    seed: int = 42
    stages: List[str] = field(default_factory=lambda: ["embed", "train", "visualize"])
    output_dir: str = MISSING
    data: DataConfig = MISSING
    embedding: EmbeddingConfig = MISSING
    model: ModelConfig = MISSING
    training: TrainingConfig = MISSING

cs = ConfigStore.instance()
cs.store(name="base_pipeline", node=PipelineConfig)
cs.store(group="embed", name="base_embed", node=EmbeddingConfig)
cs.store(group="data", name="base_data", node=DataConfig)
cs.store(group="model", name="base_mmodel", node=ModelConfig)
cs.store(group="training", name="base_train", node=TrainingConfig)

# @dataclass
# class ESMConfig(EmbeddingConfig):
#     """ESM-specific configuration"""
#     model_name: str = "esm3_sm_open_v1"
#     window_size: int = 10
#     num_windows: int = 5
#     non_contiguous: bool = True

# @dataclass
# class GearNetConfig(EmbeddingConfig):
#     """GearNet-specific configuration"""
#     hidden_dims: List[int] = field(default_factory=lambda: [512, 512, 512])
#     num_relation: int = 7
#     edge_input_dim: int = 59
#     num_angle_bin: int = 8

# @dataclass
# class Config:
#     """Main configuration"""
#     embedding: EmbeddingConfig = MISSING
#     data_dir: str = "data"
#     output_dir: str = "outputs"
#     seed: int = 42