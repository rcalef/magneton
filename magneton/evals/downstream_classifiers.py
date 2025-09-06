
import torch
import torch.nn as nn
import lightning as L
from torchmetrics import (
    Accuracy,
    AveragePrecision,
    MeanAbsoluteError,
    MeanSquaredError,
    Metric,
    MetricCollection,
    SpearmanCorrCoef,
)
from torchmetrics.functional.classification import multilabel_average_precision

from magneton.config import PipelineConfig, TrainingConfig
from magneton.data.core import Batch
from magneton.training.embedding_mlp import EmbeddingMLP

from .metrics import FMaxScore


def _get_optimizer(
    model: nn.Module,
    config: TrainingConfig,
    frozen_embedder: bool = True,
) -> torch.optim.Optimizer:
    optim_params = []
    # MLP params
    optim_params.append({
        "params": model.mlp.parameters(),
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
    })

    # Embedder params
    if not frozen_embedder:
        optim_params.append({
            "params": model.embedder.parameters(),
            "lr": config.embedding_learning_rate,
            "weight_decay": config.embedding_weight_decay,
        })
    optimizer = torch.optim.AdamW(
        optim_params,
    )
    return optimizer

class MultiLabelMLP(L.LightningModule):
    def __init__(
        self,
        config: PipelineConfig,
        task: str,
        num_classes: int,
        task_type: str = "multilabel",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self.task = task
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
        prev_dim = self.embedder.get_embed_dim()
        for hidden_dim in self.config.model.model_params["hidden_dims"]:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.config.model.model_params["dropout_rate"]),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

        # Choose loss function based on task type
        if task_type == "multilabel":
            self.loss = nn.BCEWithLogitsLoss()
        elif task_type == "multiclass":
            self.loss = nn.CrossEntropyLoss()
        elif task_type == "binary":
            self.loss = nn.BCEWithLogitsLoss()
        elif task_type == "regression":
            self.loss = nn.MSELoss()
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        # Metrics based on task type
        if task_type == "multilabel":
            self.train_metrics = MetricCollection(
                {
                    "accuracy": Accuracy(task="multilabel", num_labels=num_classes),
                    "auprc": AveragePrecision(task="multilabel", num_labels=num_classes),
                },
                prefix="train_",
            )
            self.val_metrics = self.train_metrics.clone(prefix="valid_")
            self.val_metrics.add_metrics({"fmax": FMaxScore(num_thresh_steps=101)})
        elif task_type == "multiclass":
            self.train_metrics = MetricCollection(
                {
                    "accuracy": Accuracy(task="multiclass", num_classes=num_classes),
                },
                prefix="train_",
            )
            self.val_metrics = self.train_metrics.clone(prefix="valid_")
        elif task_type == "binary":
            self.train_metrics = MetricCollection(
                {
                    "accuracy": Accuracy(task="binary"),
                    "auprc": AveragePrecision(task="binary"),
                },
                prefix="train_",
            )
            self.val_metrics = self.train_metrics.clone(prefix="valid_")
        elif task_type == "regression":
            self.train_metrics = MetricCollection(
                {
                    "mae": MeanAbsoluteError(),
                    "rmse": MeanSquaredError(squared=False),  # RMSE instead of MSE
                    "spearman": SpearmanCorrCoef(),
                },
                prefix="train_",
            )
            self.val_metrics = self.train_metrics.clone(prefix="valid_")

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

        if self.task_type == "multilabel":
            labels = batch.labels.to(dtype=logits.dtype)
        elif self.task_type == "multiclass":
            labels = batch.labels.long()  # CrossEntropy expects long integers
        elif self.task_type == "binary":
            labels = batch.labels.to(dtype=logits.dtype)
            # Ensure labels match logits shape [batch_size, 1] for BCEWithLogitsLoss
            if labels.dim() == 1:
                labels = labels.unsqueeze(-1)
        elif self.task_type == "regression":
            labels = batch.labels.to(dtype=logits.dtype)
            # Ensure labels match logits shape [batch_size, 1] for regression
            if labels.dim() == 1:
                labels = labels.unsqueeze(-1)
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

        loss = self.loss(logits, labels)

        self.log("train_loss", loss, sync_dist=True)
        if batch_idx % 50 == 0:
            if self.task_type == "multilabel":
                self.train_metrics(logits, batch.labels)
            elif self.task_type == "binary":
                # For binary metrics, convert labels to int
                int_labels = labels.squeeze().long()
                self.train_metrics(logits.squeeze(), int_labels)
            else:
                self.train_metrics(logits, labels)
            self.log_dict(self.train_metrics)

        return loss

    def validation_step(self, batch: Batch, batch_idx):
        logits = self(batch)

        if self.task_type == "multilabel":
            labels = batch.labels.to(dtype=logits.dtype)
        elif self.task_type == "multiclass":
            labels = batch.labels.long()
        elif self.task_type == "binary":
            labels = batch.labels.to(dtype=logits.dtype)
            # Ensure labels match logits shape [batch_size, 1] for BCEWithLogitsLoss
            if labels.dim() == 1:
                labels = labels.unsqueeze(-1)
        elif self.task_type == "regression":
            labels = batch.labels.to(dtype=logits.dtype)
            # Ensure labels match logits shape [batch_size, 1] for regression
            if labels.dim() == 1:
                labels = labels.unsqueeze(-1)
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

        loss = self.loss(logits, labels)

        self.log("val_loss", loss, sync_dist=True)
        if self.task_type == "multilabel":
            self.val_metrics.update(logits, batch.labels)
        elif self.task_type == "multiclass":
            self.val_metrics.update(logits, labels)
        elif self.task_type == "binary":
            # For binary metrics, convert labels to int (AveragePrecision expects int targets)
            int_labels = labels.squeeze().long()
            self.val_metrics.update(logits.squeeze(), int_labels)
        elif self.task_type == "regression":
            self.val_metrics.update(logits, labels)


    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics)
        return super().on_validation_epoch_end()

    def predict_step(self, batch: Batch, batch_idx: int, dataloader_idx: int=0):
        logits = self(batch)
        labels = batch.labels

        return logits, labels

    def configure_optimizers(self):
        return _get_optimizer(
            model=self,
            config=self.config,
            frozen_embedder=self.config.model.frozen_embedder,
        )

    def name(self) -> str:
        return f"{self.task}-mlp"

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

        # Chop off one dim bc of linear layer at end
        for hidden_dim in hidden_dims[:-1]:
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels=prev_dim,
                        out_channels=hidden_dim,
                        kernel_size=5,
                        padding="same",
                    ),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_dim = hidden_dim
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
    ):
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self.task = task
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

        self.mlp = ResidueClassifierHead(
            embed_dim=self.embedder.get_embed_dim(),
            hidden_dims=self.config.model.model_params["hidden_dims"],
            num_classes=num_classes,
            dropout_rate=self.config.model.model_params["dropout_rate"],
        )
        self.loss = nn.BCEWithLogitsLoss()

        # Metrics
        self.train_metrics = MetricCollection(
            {
                "accuracy": Accuracy(task="multilabel", num_labels=num_classes),
                "auprc": AveragePrecision(task="multilabel", num_labels=num_classes),
            },
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="valid_")
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
        return torch.cat(residue_logits)

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        logits = self(batch)

        labels = batch.labels.to(dtype=logits.dtype)
        loss = self.loss(logits, labels)

        self.log("train_loss", loss, sync_dist=True)
        if batch_idx % 50 == 0:
            self.train_metrics(logits, batch.labels)
            self.log_dict(self.train_metrics)

        return loss

    def validation_step(self, batch: Batch, batch_idx):
        logits = self(batch)

        labels = batch.labels.to(dtype=logits.dtype)
        loss = self.loss(logits, labels)

        self.log("val_loss", loss, sync_dist=True)
        self.val_metrics.update(logits, batch.labels)

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics)
        return super().on_validation_epoch_end()

    def predict_step(self, batch: Batch, batch_idx: int, dataloader_idx: int=0):
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