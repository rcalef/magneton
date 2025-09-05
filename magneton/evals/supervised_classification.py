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
    Metric,
    MetricCollection,
)
from torchmetrics.functional.classification import multilabel_average_precision

from magneton.config import PipelineConfig
from magneton.data.core import Batch
from magneton.data import SupervisedDownstreamTaskDataModule
from magneton.training.embedding_mlp import EmbeddingMLP


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
        protein_embeds = self.embedder.embed_batch(batch, protein_level=True)

        # proteins X num_classes
        return self.mlp(protein_embeds)

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

    classifier = MultiLabelMLP(
        config=config,
        task=task,
        num_classes=module.num_classes(),
    )

    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir,
            monitor="valid_auprc",
            mode="max",
            save_top_k=3,
            filename="{epoch}-{valid_auprc:.2f}",
            save_last="link",
        ),
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
    run_final_predictions(
        model=classifier,
        trainer=trainer,
        loader=module.val_dataloader(),
        num_classes=module.num_classes(),
        output_dir=output_dir,
        prefix="validation",
    )

    run_final_predictions(
        model=classifier,
        trainer=trainer,
        loader=module.test_dataloader(),
        num_classes=module.num_classes(),
        output_dir=output_dir,
        prefix="test",
    )

    if logger is not None:
        logger.experiment.finish()


def run_test_set_eval(
    config: PipelineConfig,
    task: str,
    output_dir: str,
):
    module = SupervisedDownstreamTaskDataModule(
        data_config=config.data,
        task=task,
        data_dir=config.evaluate.data_dir,
        model_type=config.embedding.model,
        distributed=False,
    )

    classifier = MultiLabelMLP.load_from_checkpoint(
        config.evaluate.model_checkpoint,
    )

    # Create trainer
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        default_root_dir=output_dir,
        precision="bf16-mixed",
    )

    output_dir = Path(output_dir)

    run_final_predictions(
        model=classifier,
        trainer=trainer,
        loader=module.test_dataloader(),
        num_classes=module.num_classes(),
        output_dir=output_dir,
        prefix="test",
    )