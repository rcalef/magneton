from pathlib import Path
from typing import Any

import lightning as L
import torch
import torch.nn as nn

from magneton.config import PipelineConfig, TrainingConfig
from magneton.data.core import Batch
from magneton.data.evals import EVAL_TASK
from magneton.training.embedding_mlp import EmbeddingMLP

from .metrics import (
    FMaxScore,
    PrecisionAtL,
    format_logits_and_labels_for_metrics,
    get_task_torchmetrics,
)


def _get_optimizer(
    model: nn.Module,
    config: TrainingConfig,
    frozen_embedder: bool = True,
) -> torch.optim.Optimizer:
    optim_params = []
    # MLP params
    optim_params.append(
        {
            "params": model.mlp.parameters(),
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


class MultiLabelMLP(L.LightningModule):
    def __init__(
        self,
        config: PipelineConfig,
        task: str,
        num_classes: int,
        task_type: EVAL_TASK = EVAL_TASK.MULTILABEL,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self.task = task
        self.task_type = task_type
        self.num_classes = num_classes

        # Load embedder
        model = EmbeddingMLP.load_from_checkpoint(
            self.config.evaluate.model_checkpoint,
            load_pretrained_fisher=self.config.evaluate.has_fisher_info,
        )
        self.embedder = model.embedder

        if self.config.model.frozen_embedder:
            self.embedder._freeze()
            self.embedder.eval()
        else:
            self.embedder._unfreeze(unfreeze_all=False)

        # Build MLP layers
        layers = []
        embed_dim = self.embedder.get_embed_dim()
        hidden_dims = parse_hidden_dims(
            raw_dims=self.config.model.model_params["hidden_dims"], embed_dim=embed_dim
        )

        prev_dim = embed_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Dropout(self.config.model.model_params["dropout_rate"]),
                    nn.Linear(prev_dim, hidden_dim),
                    nn.Tanh(),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Dropout(self.config.model.model_params["dropout_rate"]))
        layers.append(nn.Linear(prev_dim, num_classes))
        self.mlp = nn.Sequential(*layers)
        print(f"head model: {self.mlp}")

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

    def forward(
        self,
        batch: Batch,
    ) -> torch.Tensor:
        # batch_size (num_proteins)  X embed_dim
        protein_embeds = self.embedder.embed_batch(batch, protein_level=True)

        # proteins X num_classes
        return self.mlp(protein_embeds)

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        logits = self(batch)
        labels = _get_labels_for_loss(batch.labels, self.task_type, logits.dtype)

        loss = self.loss(logits, labels)

        self.log("train_loss", loss, sync_dist=True)
        if batch_idx % 50 == 0:
            logits, labels = format_logits_and_labels_for_metrics(
                logits,
                batch.labels,
                self.task_type,
            )
            self.train_metrics(logits, labels)
            self.log_dict(self.train_metrics, sync_dist=True)

        return loss

    def validation_step(self, batch: Batch, batch_idx):
        logits = self(batch)
        labels = _get_labels_for_loss(batch.labels, self.task_type, logits.dtype)

        loss = self.loss(logits, labels)

        self.log("val_loss", loss, sync_dist=True)
        logits, labels = format_logits_and_labels_for_metrics(
            logits,
            batch.labels,
            self.task_type,
        )
        self.val_metrics.update(logits, labels)

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics, sync_dist=True)
        return super().on_validation_epoch_end()

    def predict_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0):
        logits = self(batch)
        labels = batch.labels

        return logits, labels

    def configure_optimizers(self):
        return _get_optimizer(
            model=self,
            config=self.config.training,
            frozen_embedder=self.config.model.frozen_embedder,
        )

    def name(self) -> str:
        return f"{self.task}-mlp"

    def on_save_checkpoint(
        self,
        checkpoint: dict[str, Any],
    ) -> None:
        """Modify checkpointing logic to not dump the underlying embedder weights if frozen."""
        # If embedder is not frozen, then just use the default state dict with all weights
        if not self.config.model.frozen_embedder:
            return
        # Otherwise overwrite state dict with just the MLP head weights
        checkpoint["state_dict"] = self.mlp.state_dict()

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
            model.mlp.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint["state_dict"])

        return model


class ResidueClassifierHead(nn.Module):
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

    def forward(self, residue_embeds: torch.Tensor) -> torch.Tensor:
        # residue_embeds has dims (batch, max_length, embeds)

        # transpose to (batch, embeds, max_length) for CNN
        x = residue_embeds.transpose(1, 2)
        x = self.cnn(x)
        # transpose back to (batch, max_length, embeds) for final classifier
        x = x.transpose(1, 2)
        return self.classifier(x)


class ResidueClassifier(L.LightningModule):
    def __init__(
        self,
        config: PipelineConfig,
        task: str,
        num_classes: int,
        task_type: EVAL_TASK = EVAL_TASK.MULTILABEL,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self.task = task
        self.task_type = task_type
        self.num_classes = num_classes

        # Load embedder
        model = EmbeddingMLP.load_from_checkpoint(
            self.config.evaluate.model_checkpoint,
            load_pretrained_fisher=self.config.evaluate.has_fisher_info,
        )
        self.embedder = model.embedder

        if self.config.model.frozen_embedder:
            self.embedder._freeze()
            self.embedder.eval()
        else:
            self.embedder._unfreeze(unfreeze_all=False)

        embed_dim = self.embedder.get_embed_dim()
        hidden_dims = parse_hidden_dims(
            raw_dims=self.config.model.model_params["hidden_dims"],
            embed_dim=embed_dim,
        )
        self.mlp = ResidueClassifierHead(
            embed_dim=embed_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout_rate=self.config.model.model_params["dropout_rate"],
        )
        print("mlp head")
        print(self.mlp)
        self.loss = nn.BCEWithLogitsLoss()

        # Metrics based on task type
        self.train_metrics = get_task_torchmetrics(
            task_type, num_classes, prefix="train_"
        )
        self.val_metrics = self.train_metrics.clone(prefix="valid_")

        if task_type == EVAL_TASK.MULTILABEL:
            # Fmax is slightly expensive to compute, so just run on val set
            self.val_metrics.add_metrics({"fmax": FMaxScore(num_thresh_steps=101)})

    def forward(
        self,
        batch: Batch,
    ) -> torch.Tensor:
        # batch_size (num_proteins) X max_len X embed_dim
        residue_embeds = self.embedder.embed_batch(
            batch,
            protein_level=False,
            zero_non_residue_embeds=True,
        )

        # batch_size X max_len X num_classes
        raw_logits = self.mlp(residue_embeds)

        # Pull out the actual residue logits and flatten
        residue_logits = []
        for idx, length in enumerate(batch.lengths):
            residue_logits.append(raw_logits[idx, :length])

        # total_len X num_classes
        flat_logits = torch.cat(residue_logits)
        if self.task_type == EVAL_TASK.BINARY:
            # Remove the trailing dim if this is a binary classification problem
            flat_logits = flat_logits.squeeze()
        return flat_logits

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        logits = self(batch)

        labels = batch.labels.to(dtype=logits.dtype)
        loss = self.loss(logits, labels)

        logits, labels = format_logits_and_labels_for_metrics(
            logits,
            batch.labels,
            self.task_type,
        )
        self.train_metrics.update(logits, batch.labels)
        self.log("train_loss", loss, sync_dist=True)
        if batch_idx % 50 == 0:
            self.log_dict(self.train_metrics)

        return loss

    def validation_step(self, batch: Batch, batch_idx):
        logits = self(batch)

        labels = batch.labels.to(dtype=logits.dtype)
        loss = self.loss(logits, labels)

        logits, labels = format_logits_and_labels_for_metrics(
            logits,
            batch.labels,
            self.task_type,
        )
        self.log("val_loss", loss, sync_dist=True)
        self.val_metrics.update(logits, labels)

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics)
        return super().on_validation_epoch_end()

    def predict_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0):
        logits = self(batch)
        labels = batch.labels

        return logits, labels

    def configure_optimizers(self):
        return _get_optimizer(
            model=self,
            config=self.config.training,
            frozen_embedder=self.config.model.frozen_embedder,
        )

    def name(self) -> str:
        return f"{self.task}-mlp"


class ContactPredictor(L.LightningModule):
    def __init__(
        self,
        config: PipelineConfig,
        # Below arguments are for compatibility with above classifiers
        # but aren't used.
        task: str,
        num_classes: int,
        task_type: EVAL_TASK = EVAL_TASK.MULTILABEL,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.config = config

        # Load embedder
        model = EmbeddingMLP.load_from_checkpoint(
            self.config.evaluate.model_checkpoint,
            load_pretrained_fisher=self.config.evaluate.has_fisher_info,
            for_contact_prediction=True,
        )
        self.embedder = model.embedder

        if not self.config.model.frozen_embedder:
            raise ValueError("contact prediction canonically uses frozen embedders")
        self.embedder._freeze()

        layers = []
        input_dim = self.embedder.get_attention_dim()
        hidden_dims = parse_hidden_dims(
            raw_dims=self.config.model.model_params["hidden_dims"], embed_dim=input_dim
        )
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
        print(f"contact head model: {self.contact_head}")

        self.loss = nn.BCEWithLogitsLoss()

        # Metrics based on task type
        self.train_metrics = get_task_torchmetrics(EVAL_TASK.BINARY, 1, prefix="train_")
        self.val_metrics = self.train_metrics.clone(prefix="valid_")
        self.p_at_l = PrecisionAtL()

    def forward(
        self,
        batch: Batch,
    ) -> torch.Tensor:
        with torch.inference_mode():
            contact_inputs = self.embedder.forward_for_contact(batch)

        # These come out of the contact head as (batch, seq_len, seq_len, 1),
        # so squeeze off the last dim.
        return self.contact_head(contact_inputs).squeeze()

    def _calc_loss(
        self,
        batch: Batch,
    ) -> tuple[torch.Tensor]:
        contact_logits = self(batch)
        labels = batch.labels

        contact_logits_flat = contact_logits.view(-1)
        labels_flat = labels.view(-1)
        keep_idxs = labels_flat != -1

        want_logits = contact_logits_flat[keep_idxs]
        want_labels = labels_flat[keep_idxs]

        loss = self.loss(want_logits, want_labels)
        return loss, contact_logits, want_logits, want_labels

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        loss, logits, flat_logits, flat_labels = self._calc_loss(batch)

        self.log("train_loss", loss, sync_dist=True)

        logits, labels = format_logits_and_labels_for_metrics(
            flat_logits,
            flat_labels,
            EVAL_TASK.BINARY,
        )
        self.train_metrics.update(logits, labels)
        if batch_idx % 50 == 0:
            self.log_dict(self.train_metrics, sync_dist=True)

        return loss

    def validation_step(self, batch: Batch, batch_idx):
        loss, logits, flat_logits, flat_labels = self._calc_loss(batch)

        flat_logits, flat_labels = format_logits_and_labels_for_metrics(
            flat_logits,
            flat_labels,
            EVAL_TASK.BINARY,
        )
        self.log("val_loss", loss, sync_dist=True)
        self.val_metrics.update(flat_logits, flat_labels)
        self.p_at_l.update(logits, batch.labels, batch.lengths)

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics, sync_dist=True)
        self.log_dict(self.p_at_l.compute(), sync_dist=True)
        return super().on_validation_epoch_end()

    def predict_step(
        self, batch: Batch, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        logits = self(batch)

        return logits

    def configure_optimizers(self):
        # Embedder params
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.contact_head.parameters(),
                    "lr": self.config.training.learning_rate,
                    "weight_decay": self.config.training.weight_decay,
                    "betas": (0.9, 0.98),
                }
            ]
        )
        return optimizer

    def name(self) -> str:
        return f"{self.task}-contact-pred"

    def on_save_checkpoint(
        self,
        checkpoint: dict[str, Any],
    ) -> None:
        """Modify checkpointing logic to not dump the underlying embedder weights if frozen."""
        # If embedder is not frozen, then just use the default state dict with all weights
        if not self.config.model.frozen_embedder:
            return
        # Otherwise overwrite state dict with just the MLP head weights
        checkpoint["state_dict"] = self.contact_head.state_dict()

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
            model.contact_head.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint["state_dict"])

        return model


class PPIPredictor(L.LightningModule):
    def __init__(
        self,
        config: PipelineConfig,
        # Below arguments are for compatibility with above classifiers
        # but aren't used.
        task: str,
        num_classes: int,
        task_type: EVAL_TASK = EVAL_TASK.BINARY,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.config = config

        # Load embedder
        model = EmbeddingMLP.load_from_checkpoint(
            self.config.evaluate.model_checkpoint,
            load_pretrained_fisher=self.config.evaluate.has_fisher_info,
        )
        self.embedder = model.embedder

        if self.config.model.frozen_embedder:
            self.embedder._freeze()
            self.embedder.eval()
        else:
            self.embedder._unfreeze(unfreeze_all=False)

        # Build MLP layers
        layers = []

        # Embed dim is here is 2 x model_embed_dim, since we just concatenate
        # the embeddings of the two proteins for input to the predictor head
        embed_dim = self.embedder.get_embed_dim() * 2
        hidden_dims = parse_hidden_dims(
            raw_dims=self.config.model.model_params["hidden_dims"], embed_dim=embed_dim
        )

        prev_dim = embed_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Dropout(self.config.model.model_params["dropout_rate"]),
                    nn.Linear(prev_dim, hidden_dim),
                    nn.Tanh(),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Dropout(self.config.model.model_params["dropout_rate"]))
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)
        print(f"head model: {self.mlp}")

        self.loss = nn.BCEWithLogitsLoss()

        # Metrics based on task type
        self.train_metrics = get_task_torchmetrics(
            EVAL_TASK.BINARY, num_classes, prefix="train_"
        )
        self.val_metrics = self.train_metrics.clone(prefix="valid_")

    def forward(
        self,
        batch: Batch,
    ) -> torch.Tensor:
        assert len(batch.protein_ids) % 2 == 0, f"got: {batch}"
        # batch_size (num_proteins)  X embed_dim
        protein_embeds = self.embedder.embed_batch(batch, protein_level=True)

        # Each pair of adjacent elements is one PPI pair, so want to
        # concatenate those
        num_proteins, embed_dim = protein_embeds.shape
        num_ppis = num_proteins // 2

        # (num_ppis, 2*embed_dim)
        ppi_embeds = protein_embeds.view(num_ppis, 2, embed_dim).flatten(1)

        # (num_ppis), squeeze off trailing dim
        return self.mlp(ppi_embeds).squeeze()

    def _collapse_ppi_labels(
        self,
        batch: Batch,
    ) -> torch.Tensor:
        # batch has rows for each protein, each pair of adjacent rows
        # is a PPI pair. Each pair of labels should be the same. Want
        # to collapse to one label per PPI pair
        assert len(batch.labels) % 2 == 0, f"got: {batch}"
        labels = batch.labels[::2]
        labels_check = batch.labels[1::2]
        assert (labels == labels_check).all(), f"got: {batch}"
        return labels

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        labels = self._collapse_ppi_labels(batch)
        logits = self(batch)

        # labels = _get_labels_for_loss(batch.labels, self.task_type, logits.dtype)

        loss = self.loss(logits, labels.float())

        self.log("train_loss", loss, sync_dist=True)
        if batch_idx % 50 == 0:
            logits, labels = format_logits_and_labels_for_metrics(
                logits,
                labels,
                EVAL_TASK.BINARY,
            )
            self.train_metrics(logits, labels)
            self.log_dict(self.train_metrics, sync_dist=True)

        return loss

    def validation_step(self, batch: Batch, batch_idx):
        labels = self._collapse_ppi_labels(batch)
        logits = self(batch)
        # labels = _get_labels_for_loss(batch.labels, self.task_type, logits.dtype)

        loss = self.loss(logits, labels.float())

        self.log("val_loss", loss, sync_dist=True)
        logits, labels = format_logits_and_labels_for_metrics(
            logits,
            labels,
            EVAL_TASK.BINARY,
        )
        self.val_metrics.update(logits, labels)

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics, sync_dist=True)
        return super().on_validation_epoch_end()

    def predict_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0):
        labels = self._collapse_ppi_labels(batch)
        logits = self(batch)

        return logits, labels

    def configure_optimizers(self):
        return _get_optimizer(
            model=self,
            config=self.config.training,
            frozen_embedder=self.config.model.frozen_embedder,
        )

    def name(self) -> str:
        return f"{self.task}-mlp"

    def on_save_checkpoint(
        self,
        checkpoint: dict[str, Any],
    ) -> None:
        """Modify checkpointing logic to not dump the underlying embedder weights if frozen."""
        # If embedder is not frozen, then just use the default state dict with all weights
        if not self.config.model.frozen_embedder:
            return
        # Otherwise overwrite state dict with just the MLP head weights
        checkpoint["state_dict"] = self.mlp.state_dict()

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
            model.mlp.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint["state_dict"])

        return model
