from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import lightning as L
import torch
import torch.nn as nn

from magneton.config import PipelineConfig, TrainingConfig
from magneton.data.core import Batch
from magneton.data.evals import EVAL_TASK, TASK_GRANULARITY
from magneton.training.embedding_mlp import EmbeddingMLP

from .metrics import (
    FMaxScore,
    PrecisionAtL,
    format_logits_and_labels_for_metrics,
    get_task_torchmetrics,
)


def parse_hidden_dims(
    raw_dims: list[int | str],
    embed_dim: int,
) -> list[int]:
    parsed_dims = []
    for dim in raw_dims:
        try:
            dim = int(dim)
        except ValueError:
            if dim != "embed":
                raise ValueError(f"unknown hidden dim: {dim}")
            else:
                dim = embed_dim
        parsed_dims.append(dim)
    return parsed_dims


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


class HeadModule(ABC):
    """Abstract base class for task-specific head modules."""

    @abstractmethod
    def forward(self, batch: Batch, embedder: nn.Module) -> torch.Tensor:
        """Forward pass through the head module."""
        pass

    def process_labels(self, batch: Batch, logits: torch.Tensor) -> torch.Tensor:
        """Implement any task-specific label processing.

        Args:
            batch: The batch of data.
            logits: The logits from the head module.

        Returns:
            The processed labels.
        This is used for more bespoke label processing, e.g. deduplicating labels
        for protein pairs in PPI, or flattening the labels for contact prediction.
        Processing for different task types (e.g. binary vs multiclass) is handled
        separately.
        """
        return batch.labels

    @abstractmethod
    def get_embed_dim(self, embedder: nn.Module) -> int:
        """Get the required embedding dimension for this head."""
        pass


class ProteinClassificationHead(nn.Module, HeadModule):
    """Head module for protein-level classification tasks."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dims: list[int],
        num_classes: int,
        dropout_rate: float,
    ):
        super().__init__()
        # Build MLP layers
        layers = []
        prev_dim = embed_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Dropout(dropout_rate),
                    nn.Linear(prev_dim, hidden_dim),
                    nn.Tanh(),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(prev_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, batch: Batch, embedder: nn.Module) -> torch.Tensor:
        # batch_size (num_proteins) X embed_dim
        protein_embeds = embedder.embed_batch(batch, protein_level=True)
        # proteins X num_classes
        return self.mlp(protein_embeds)

    def get_embed_dim(self, embedder: nn.Module) -> int:
        return embedder.get_embed_dim()


class ResidueClassificationHead(nn.Module, HeadModule):
    """Head module for residue-level classification tasks."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dims: list[int],
        num_classes: int,
        dropout_rate: float,
    ):
        super().__init__()
        # Build CNN layers
        layers = []
        prev_dim = embed_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Dropout(dropout_rate),
                    nn.Conv1d(
                        in_channels=prev_dim,
                        out_channels=hidden_dim,
                        kernel_size=5,
                        padding="same",
                    ),
                    nn.Tanh(),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Dropout(dropout_rate))
        self.cnn = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(self, batch: Batch, embedder: nn.Module) -> torch.Tensor:
        # batch_size (num_proteins) X max_len X embed_dim
        residue_embeds = embedder.embed_batch(
            batch,
            protein_level=False,
            zero_non_residue_embeds=True,
        )

        # batch_size X max_len X num_classes
        raw_logits = self._forward_cnn(residue_embeds)

        # Pull out the actual residue logits and flatten
        residue_logits = []
        for idx, length in enumerate(batch.lengths):
            residue_logits.append(raw_logits[idx, :length])

        # total_len X num_classes
        flat_logits = torch.cat(residue_logits)
        return flat_logits

    def _forward_cnn(self, residue_embeds: torch.Tensor) -> torch.Tensor:
        # residue_embeds has dims (batch, max_length, embeds)
        # transpose to (batch, embeds, max_length) for CNN
        x = residue_embeds.transpose(1, 2)
        x = self.cnn(x)
        # transpose back to (batch, max_length, embeds) for final classifier
        x = x.transpose(1, 2)
        return self.classifier(x)

    def get_embed_dim(self, embedder: nn.Module) -> int:
        return embedder.get_embed_dim()


class ContactPredictionHead(nn.Module, HeadModule):
    """Head module for contact prediction tasks.
    Args:
        input_dim: The dimension of the inputs, e.g. could be
            embedding dim or stacked attention weights from base model.
        hidden_dims: The dimensions of the hidden layers.
        ignore_index: The index to ignore in the labels.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        ignore_index: int = -1,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.Tanh(),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.contact_head = nn.Sequential(*layers)
        self.ignore_index = ignore_index

    def forward(self, batch: Batch, embedder: nn.Module) -> torch.Tensor:
        with torch.no_grad():
            contact_inputs = embedder.forward_for_contact(batch)

        # These come out of the contact head as (batch, seq_len, seq_len, 1),
        # so squeeze off the last dim.
        return self.contact_head(contact_inputs).squeeze(-1)

    def process_labels(self, batch: Batch, logits: torch.Tensor) -> torch.Tensor:
        # Contact prediction has special label processing
        labels_flat = batch.labels.view(-1)
        keep_idxs = labels_flat != self.ignore_index
        return labels_flat[keep_idxs]

    def get_embed_dim(self, embedder: nn.Module) -> int:
        return embedder.get_attention_dim()


class PPIPredictionHead(nn.Module, HeadModule):
    """Head module for protein-protein interaction prediction tasks."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dims: list[int],
        dropout_rate: float,
    ):
        super().__init__()
        # Embed dim is here is 2 x model_embed_dim, since we just concatenate
        # the embeddings of the two proteins for input to the predictor head
        input_dim = embed_dim * 2

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Dropout(dropout_rate),
                    nn.Linear(prev_dim, hidden_dim),
                    nn.Tanh(),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, batch: Batch, embedder: nn.Module) -> torch.Tensor:
        assert len(batch.protein_ids) % 2 == 0, f"got: {batch}"
        # batch_size (num_proteins) X embed_dim
        protein_embeds = embedder.embed_batch(batch, protein_level=True)

        # Each pair of adjacent elements is one PPI pair, so want to
        # concatenate those
        num_proteins, embed_dim = protein_embeds.shape
        num_ppis = num_proteins // 2

        # (num_ppis, 2*embed_dim)
        ppi_embeds = protein_embeds.view(num_ppis, 2, embed_dim).flatten(1)

        # (num_ppis, 1)
        return self.mlp(ppi_embeds)

    def process_labels(self, batch: Batch, logits: torch.Tensor) -> torch.Tensor:
        # PPI prediction has special label processing
        assert len(batch.labels) % 2 == 0, f"got: {batch}"
        labels = batch.labels[::2]
        labels_check = batch.labels[1::2]
        assert (labels == labels_check).all(), f"got: {batch}"

        return labels

    def get_embed_dim(self, embedder: nn.Module) -> int:
        return embedder.get_embed_dim()


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
        model = EmbeddingMLP.load_from_checkpoint(
            self.config.evaluate.model_checkpoint,
            load_pretrained_fisher=self.config.evaluate.has_fisher_info,
            for_contact_prediction=(
                task_granularity == TASK_GRANULARITY.CONTACT_PREDICTION
            ),
        )
        self.embedder = model.embedder

        if self.config.model.frozen_embedder:
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

        self.log("train_loss", loss, sync_dist=True)
        if batch_idx % 50 == 0:
            logits, labels = format_logits_and_labels_for_metrics(
                logits,
                labels,
                self.task_type,
            )
            self.train_metrics(logits, labels)
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
            frozen_embedder=self.config.model.frozen_embedder,
        )

    def name(self) -> str:
        return f"{self.task}-{self.task_granularity.value}"

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Modify checkpointing logic to not dump the underlying embedder weights if frozen."""
        # If embedder is not frozen, then just use the default state dict with all weights
        if not self.config.model.frozen_embedder:
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
