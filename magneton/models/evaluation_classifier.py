from pathlib import Path
from typing import Any

import lightning as L
import torch
import torch.nn as nn

from magneton.config import PipelineConfig, TrainingConfig
from magneton.data.core import Batch
from magneton.data.evaluations import EVAL_TASK, TASK_GRANULARITY
from magneton.evaluations.metrics import (
    FMaxScore,
    PrecisionAtL,
    format_logits_and_labels_for_metrics,
    get_task_torchmetrics,
)

from .head_classifiers import (
    ContactPredictionHead,
    HeadModule,
    PPIPredictionHead,
    ProteinClassificationHead,
    ResidueClassificationHead,
)
from .substructure_classifier import SubstructureClassifier
from .utils import parse_hidden_dims


class EvaluationClassifier(L.LightningModule):
    """Unified LightningModule for all evaluation tasks."""

    def __init__(
        self,
        config: PipelineConfig,
        task: str,
        num_classes: int,
        task_type: EVAL_TASK,
        task_granularity: TASK_GRANULARITY,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self.task = task
        self.task_type = task_type
        self.task_granularity = task_granularity
        self.num_classes = num_classes

        # Load embedder
        model = SubstructureClassifier.load_from_checkpoint(
            self.config.evaluate.model_checkpoint,
            load_pretrained_fisher=self.config.evaluate.has_fisher_info,
            for_contact_prediction=(
                task_granularity == TASK_GRANULARITY.CONTACT_PREDICTION
            ),
        )
        self.embedder = model.embedder

        if self.config.model.frozen_base_model:
            self.embedder._freeze()
            self.embedder.eval()
        else:
            self.embedder._unfreeze(unfreeze_all=False)

        # Create head module
        embed_dim = self.embedder.get_embed_dim()
        if task_granularity == TASK_GRANULARITY.CONTACT_PREDICTION:
            embed_dim = self.embedder.get_attention_dim()
        elif task_granularity == TASK_GRANULARITY.PPI_PREDICTION:
            # PPI head will handle the 2x concatenation internally
            pass

        hidden_dims = parse_hidden_dims(
            raw_dims=self.config.model.model_params["hidden_dims"], embed_dim=embed_dim
        )

        self.head = _create_head_module(
            task_granularity=task_granularity,
            embed_dim=embed_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout_rate=self.config.model.model_params["dropout_rate"],
        )

        print(f"head model: {self.head}")

        # Choose loss function based on task type
        if task_type == EVAL_TASK.MULTILABEL:
            self.loss = nn.BCEWithLogitsLoss()
        elif task_type == EVAL_TASK.MULTICLASS:
            self.loss = nn.CrossEntropyLoss()
        elif task_type == EVAL_TASK.BINARY:
            self.loss = nn.BCEWithLogitsLoss()
        elif task_type == EVAL_TASK.REGRESSION:
            self.loss = nn.MSELoss()
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        # Metrics based on task type
        self.train_metrics = get_task_torchmetrics(
            task_type, num_classes, prefix="train_"
        )
        self.val_metrics = self.train_metrics.clone(prefix="valid_")

        if task_type == EVAL_TASK.MULTILABEL:
            # Fmax is slightly expensive to compute, so just run on val set
            self.val_metrics.add_metrics({"fmax": FMaxScore(num_thresh_steps=101)})

        # Special metrics for contact prediction
        if task_granularity == TASK_GRANULARITY.CONTACT_PREDICTION:
            self.p_at_l = PrecisionAtL()

    def forward(self, batch: Batch) -> torch.Tensor:
        """Forward pass through the model."""
        return self.head.forward(batch, self.embedder)

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        logits = self(batch)

        # Perform any task-specific label processing as well as any
        # reshaping for the general type of eval task.
        labels = self.head.process_labels(batch, logits)
        labels = _get_labels_for_loss(labels, self.task_type, logits.dtype)

        loss = self.loss(logits, labels)

        self.log("train_loss", loss, sync_dist=False)

        logits, labels = format_logits_and_labels_for_metrics(
            logits,
            labels,
            self.task_type,
        )
        self.train_metrics.update(logits, labels)

        if batch_idx % 50 == 0:
            self.log_dict(self.train_metrics, sync_dist=True)
        return loss

    def validation_step(self, batch: Batch, batch_idx):
        logits = self(batch)

        labels = self.head.process_labels(batch, logits)
        labels = _get_labels_for_loss(labels, self.task_type, logits.dtype)

        loss = self.loss(logits, labels)

        self.log("val_loss", loss, sync_dist=True)
        logits, labels = format_logits_and_labels_for_metrics(
            logits,
            labels,
            self.task_type,
        )
        self.val_metrics.update(logits, labels)

        # Special handling for contact prediction
        if self.task_granularity == TASK_GRANULARITY.CONTACT_PREDICTION:
            self.p_at_l.update(logits, batch.labels, batch.lengths)

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics, sync_dist=True)
        if self.task_granularity == TASK_GRANULARITY.CONTACT_PREDICTION:
            self.log_dict(self.p_at_l.compute(), sync_dist=True)
        return super().on_validation_epoch_end()

    def predict_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0):
        logits = self(batch)

        labels = self.head.process_labels(batch, logits)
        labels = _get_labels_for_loss(labels, self.task_type, logits.dtype)

        return logits, labels

    def configure_optimizers(self):
        return _get_optimizer(
            model=self,
            config=self.config.training,
            frozen_embedder=self.config.model.frozen_base_model,
        )

    def name(self) -> str:
        return f"{self.task}-{self.task_granularity.value}"

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Modify checkpointing logic to not dump the underlying embedder weights if frozen."""
        # If embedder is not frozen, then just use the default state dict with all weights
        if not self.config.model.frozen_base_model:
            return
        # Otherwise overwrite state dict with just the head weights
        checkpoint["state_dict"] = self.head.state_dict()

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Path,
        **kwargs,
    ) -> None:
        """Modify checkpointing logic to not dump the underlying embedder weights if frozen."""
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        hparams = checkpoint["hyper_parameters"]
        hparams.update(kwargs)

        model = cls(**hparams)

        if hparams["config"].model.frozen_embedder:
            model.head.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint["state_dict"])

        return model


def _get_labels_for_loss(
    labels: torch.Tensor,
    task_type: EVAL_TASK,
    logits_dtype: torch.dtype,
) -> torch.Tensor:
    if task_type == EVAL_TASK.MULTILABEL:
        labels = labels.to(dtype=logits_dtype)
    elif task_type == EVAL_TASK.MULTICLASS:
        labels = labels.long()  # CrossEntropy expects long integers
    elif task_type == EVAL_TASK.BINARY:
        labels = labels.to(dtype=logits_dtype)
        # Ensure labels match logits shape [batch_size, 1] for BCEWithLogitsLoss
        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)
    elif task_type == EVAL_TASK.REGRESSION:
        labels = labels.to(dtype=logits_dtype)
        # Ensure labels match logits shape [batch_size, 1] for regression
        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    return labels


def _create_head_module(
    task_granularity: TASK_GRANULARITY,
    embed_dim: int,
    hidden_dims: list[int],
    num_classes: int,
    dropout_rate: float,
) -> HeadModule:
    """Factory function to create the appropriate head module based on task granularity."""
    if task_granularity == TASK_GRANULARITY.PROTEIN_CLASSIFICATION:
        return ProteinClassificationHead(
            embed_dim=embed_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )
    elif task_granularity == TASK_GRANULARITY.RESIDUE_CLASSIFICATION:
        return ResidueClassificationHead(
            embed_dim=embed_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )
    elif task_granularity == TASK_GRANULARITY.CONTACT_PREDICTION:
        return ContactPredictionHead(
            input_dim=embed_dim,
            hidden_dims=hidden_dims,
        )
    elif task_granularity == TASK_GRANULARITY.PPI_PREDICTION:
        return PPIPredictionHead(
            embed_dim=embed_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
        )
    else:
        raise ValueError(f"Unknown task granularity: {task_granularity}")


def _get_optimizer(
    model: EvaluationClassifier,
    config: TrainingConfig,
    frozen_embedder: bool = True,
) -> torch.optim.Optimizer:
    optim_params = []
    # MLP params
    optim_params.append(
        {
            "params": model.head.parameters(),
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay,
            "betas": (0.9, 0.98),
        }
    )

    # Embedder params
    if not frozen_embedder:
        optim_params.append(
            {
                "params": model.embedder.parameters(),
                "lr": config.embedding_learning_rate,
                "weight_decay": config.embedding_weight_decay,
                "betas": (0.9, 0.98),
            }
        )
    optimizer = torch.optim.AdamW(
        optim_params,
    )
    return optimizer
