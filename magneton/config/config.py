from dataclasses import dataclass, field
from typing import List, Optional
from omegaconf import MISSING

@dataclass
class EmbeddingConfig:
    max_seq_length: int = MISSING
    batch_size: int = 32
    device: str = "cuda"
    model_params: dict = MISSING

@dataclass
class DataConfig:
    data_dir: str = MISSING

@dataclass
class ModelConfig:
    model_type: str = MISSING

@dataclass
class TrainingConfig:
    max_epochs: int = 100

@dataclass
class PipelineConfig:
    seed: int = 42
    output_dir: str = MISSING
    data: DataConfig = MISSING
    embedding: EmbeddingConfig = MISSING
    model: ModelConfig = MISSING
    training: TrainingConfig = MISSING

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