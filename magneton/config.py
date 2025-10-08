from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

@dataclass
class EmbeddingConfig:
    _target_: str = "magneton.config.EmbeddingConfig"
    model: str = MISSING
    model_params: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class EvalConfig:
    _target_: str = "magneton.config.EvalConfig"
    tasks: List[str] = MISSING
    data_dir: str = MISSING
    model_checkpoint: str = MISSING
    final_prediction_only: bool = False
    rerun_completed: bool = False
    # Whether or not the saved model used EWC loss
    has_fisher_info: bool = False


@dataclass
class DataConfig:
    _target_: str = "magneton.config.DataConfig"
    data_dir: str = MISSING
    compression: str = "gz"
    prefix: str = "swissprot.with_ss"
    splits: str = MISSING
    batch_size: int = 32
    fasta_path: Optional[str] = None
    labels_path: Optional[str] = None
    struct_template: Optional[str] = None
    substruct_types: Optional[List[str]] = None
    collapse_labels: bool = True
    num_loader_workers: int = 4
    model_specific_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    _target_: str = "magneton.config.ModelConfig"
    model_type: str = MISSING
    model_params: Optional[Dict[str, Any]] = field(default_factory=dict)
    checkpoint: str = MISSING
    frozen_embedder: bool = True


@dataclass
class TrainingConfig:
    _target_: str = "magneton.config.TrainingConfig"
    max_epochs: int = 100
    learning_rate: float = 1e-2
    weight_decay: float = 0.0
    embedding_learning_rate: float = 1e-5
    embedding_weight_decay: float = 0.0
    accelerator: str = "gpu"
    strategy: str = "ddp"
    precision: str = "bf16-mixed"
    accumulate_grad_batches: int = 1
    devices: Optional[Any] = "auto"
    additional_training_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)
    # This is supposed to be Union[int, bool], but that causes unsolved assertion errors. See:
    #   https://github.com/facebookresearch/hydra/issues/2738
    dev_run: Any = False
    profile: bool = False
    loss_strategy: str = "standard"
    ewc_weight: float = 400
    reuse_ewc_weights: Optional[str] = None


@dataclass
class PipelineConfig:
    _target_: str = "magneton.config.PipelineConfig"
    seed: int = 42
    stage: str = "train"
    output_dir: str = MISSING
    run_id: str = MISSING
    data: DataConfig = MISSING
    embedding: EmbeddingConfig = MISSING
    model: ModelConfig = MISSING
    training: Optional[TrainingConfig] = None
    evaluate: Optional[EvalConfig] = None


cs = ConfigStore.instance()
cs.store(name="base_pipeline", node=PipelineConfig)
cs.store(group="embed", name="base_embed", node=EmbeddingConfig)
cs.store(group="data", name="base_data", node=DataConfig)
cs.store(group="model", name="base_mmodel", node=ModelConfig)
cs.store(group="training", name="base_train", node=TrainingConfig)
