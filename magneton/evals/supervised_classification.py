from pathlib import Path

import torch
import torch.nn as nn
import lightning as L

from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import (
    CSVLogger,
    WandbLogger,
)
from torchdata.nodes import Loader
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

from magneton.config import PipelineConfig
from magneton.data.core import Batch
from magneton.data import SupervisedDownstreamTaskDataModule
from magneton.embedders.base_embedder import BaseEmbedder
from magneton.training.embedding_mlp import EmbeddingMLP, MultitaskEmbeddingMLP


def _calc_fmax(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_thresh_steps: int=101,
) -> torch.Tensor:
    probs = logits.sigmoid()
    f1s = []
    for thresh in torch.linspace(start=0, end=1, steps=num_thresh_steps):
        preds = probs >= thresh

        tp = ((preds == labels) & labels).sum()
        fp = ((preds != labels) & labels).sum()
        tn = ((preds == labels) & ~labels).sum()
        fn = ((preds != labels) & ~labels).sum()

        f1 = (2*tp) / (2*tp + fp + fn)
        f1s.append(f1)

    f1s = torch.stack(f1s)
    return f1s.max()


class FMaxScore(Metric):
    def __init__(self, **kwargs):
        self.num_thresh_steps = kwargs.pop("num_thresh_steps", 101)
        super().__init__(**kwargs)

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("labels", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        self.preds.append(preds.detach().cpu())
        self.labels.append(target.detach().cpu())

    def compute(self) -> torch.Tensor:
        all_preds = torch.cat(self.preds)
        all_labels = torch.cat(self.labels)

        return _calc_fmax(all_preds, all_labels)

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
        # batch_size (num_proteins) X max_len X embed_dim
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
            self.val_metrics.update(logits, labels)
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
        optim_params = []
        # MLP params
        optim_params.append({
            "params": self.mlp.parameters(),
            "lr": self.config.training.learning_rate,
            "weight_decay": self.config.training.weight_decay,
        })

        # Embedder params
        if not self.config.model.frozen_embedder:
            optim_params.append({
                "params": self.embedder.parameters(),
                "lr": self.config.training.embedding_learning_rate,
                "weight_decay": self.config.training.embedding_weight_decay,
            })
        optimizer = torch.optim.AdamW(
            optim_params,
        )
        return optimizer

    def name(self) -> str:
        return f"{self.task}-mlp"

def run_final_predictions(
    model: MultiLabelMLP,
    trainer: L.Trainer,
    loader: Loader,
    num_classes: int,
    output_dir: Path,
    prefix: str,
):
    final_predictions = trainer.predict(
        model=model,
        dataloaders=loader,
        return_predictions=True
    )

    all_logits, all_labels = zip(*final_predictions)
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    # Make another pass through dataset to collect IDs
    protein_ids = []
    for batch in loader:
        protein_ids.extend(batch.protein_ids)

    results_dict = {
        "protein_ids": protein_ids,
        "logits": all_logits,
        "labels": all_labels,
    }

    torch.save(results_dict, output_dir / f"{prefix}_results.pt")

    final_fmax = _calc_fmax(all_logits, all_labels)
    final_auprc = multilabel_average_precision(
        all_logits,
        all_labels,
        num_labels=num_classes,
    )
    print(f"{prefix} final Fmax: {final_fmax.item():0.3f}")
    print(f"{prefix} final AUPRC: {final_auprc.item():0.3f}")


def run_supervised_classification(
    config: PipelineConfig,
    task: str,
    output_dir: str,
    run_id: str,
):
    module = SupervisedDownstreamTaskDataModule(
        data_config=config.data,
        task=task,
        data_dir=config.evaluate.data_dir,
        model_type=config.embedding.model,
        distributed=False,
    )

    # Determine task type based on the task name
    task_type = "multilabel"  # Default for non-PEER tasks
    
    # PEER multiclass classification tasks
    if task in ["fold", "subcellular_localization"]:
        task_type = "multiclass"
    # PEER binary classification tasks  
    elif task in ["solubility", "binary_localization"]:
        task_type = "binary" 
    # PEER regression tasks (single sequence)
    elif task in ["fluorescence", "stability", "beta_lactamase", "aav", "gb1", "thermostability"]:
        task_type = "regression"
    
    classifier = MultiLabelMLP(
        config=config,
        task=task,
        num_classes=module.num_classes(),
        task_type=task_type,
    )

    # Set up callbacks with appropriate monitoring metric
    if task_type == "multiclass":
        monitor_metric = "valid_accuracy"
        mode = "max"
        filename = "{epoch}-{valid_accuracy:.2f}"
    elif task_type == "binary":
        monitor_metric = "valid_auprc"
        mode = "max"
        filename = "{epoch}-{valid_auprc:.2f}"
    elif task_type == "regression":
        monitor_metric = "valid_spearman"
        mode = "max"
        filename = "{epoch}-{valid_spearman:.2f}"
    else:  # multilabel
        monitor_metric = "valid_auprc"
        mode = "max"
        filename = "{epoch}-{valid_auprc:.2f}"
    
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir,
            monitor=monitor_metric,
            mode=mode,
            save_top_k=3,
            filename=filename
        ),
        # EarlyStopping(
        #     monitor=monitor_metric,
        #     mode=mode,
        #     patience=4,
        # ),
    ]
    if config.training.dev_run is not None:
        logger = None
    else:
        logger = WandbLogger(
            entity="magneton",
            project="magneton",
            name=run_id,
        )

    # Create trainer
    trainer = L.Trainer(
        strategy="auto",
        callbacks=callbacks,
        logger=logger,
        accelerator="gpu",
        devices="auto",
        default_root_dir=output_dir,
        max_epochs=config.training.max_epochs,
        precision="bf16-mixed",
        val_check_interval=1.0,
    )
    trainer.fit(
        classifier,
        datamodule=module,
    )

    output_dir = Path(output_dir)
    
    # Run final predictions
    final_predictions = trainer.predict(
        model=classifier,
        dataloaders=module.val_dataloader(),
        return_predictions=True
    )

    all_logits, all_labels = zip(*final_predictions)
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    torch.save(all_logits, output_dir / "val_logits.pt")
    torch.save(all_labels, output_dir / "val_labels.pt")

    # Calculate task-specific metrics
    if task_type == "multilabel":
        final_fmax = _calc_fmax(all_logits, all_labels)
        final_auprc = multilabel_average_precision(
            all_logits,
            all_labels,
            num_labels=module.num_classes(),
        )
        print(f"Final Fmax: {final_fmax.item():0.3f}")
        print(f"Final AUPRC: {final_auprc.item():0.3f}")
    elif task_type == "multiclass":
        # For multiclass, calculate accuracy
        predicted_classes = all_logits.argmax(dim=1)
        accuracy = (predicted_classes == all_labels).float().mean()
        print(f"Final Accuracy: {accuracy.item():0.3f}")
    elif task_type == "binary":
        # For binary classification
        predicted_probs = all_logits.sigmoid()
        predicted_classes = (predicted_probs >= 0.5).float()
        accuracy = (predicted_classes == all_labels).float().mean()
        auprc = AveragePrecision(task="binary")(all_logits, all_labels.long())
        print(f"Final Accuracy: {accuracy.item():0.3f}")
        print(f"Final AUPRC: {auprc.item():0.3f}")
    elif task_type == "regression":
        # For regression tasks - ensure shape compatibility
        if all_labels.dim() == 1:
            all_labels = all_labels.unsqueeze(-1)  # Match logits shape [N, 1]
        
        mae = MeanAbsoluteError()(all_logits, all_labels)
        rmse = MeanSquaredError(squared=False)(all_logits, all_labels)
        spearman = SpearmanCorrCoef()(all_logits.squeeze(), all_labels.squeeze())
        print(f"Final MAE: {mae.item():0.3f}")
        print(f"Final RMSE: {rmse.item():0.3f}")
        print(f"Final Spearman: {spearman.item():0.3f}")
    else:
        print(f"No final metrics defined for task_type: {task_type}")

    if logger is not None:
        logger.experiment.finish()
