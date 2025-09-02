from pathlib import Path

import torch
import torch.nn as nn
import lightning as L

from esm.tokenization import get_esmc_model_tokenizers
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import (
    CSVLogger,
    WandbLogger,
)
from torchmetrics import (
    Accuracy,
    AveragePrecision,
    Metric,
    MetricCollection,
)
from torchmetrics.functional.classification import multilabel_average_precision

from magneton.config import DataConfig, EvalConfig
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
        embedder: BaseEmbedder,
        task: str,
        num_classes: int,
        config: EvalConfig,
    ):
        super().__init__()
        self.config = config
        self.task = task
        self.num_classes = num_classes
        self.embedder = embedder

        self.embedder._freeze()
        self.embedder.eval()

        # Build MLP layers
        layers = []
        prev_dim = self.embedder.get_embed_dim()
        for hidden_dim in self.config.model_params["hidden_dims"]:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.config.model_params["dropout_rate"]),
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
            "lr": self.config.model_params["learning_rate"],
            "weight_decay": self.config.model_params["weight_decay"],
        })
        optimizer = torch.optim.AdamW(
            optim_params,
        )
        return optimizer

    def name(self) -> str:
        return f"{self.task}-mlp"

def run_supervised_classification(
    model: EmbeddingMLP | MultitaskEmbeddingMLP,
    task: str,
    output_dir: str,
    run_id: str,
    eval_config: EvalConfig,
    data_config: DataConfig,
):
    module = SupervisedDownstreamTaskDataModule(
        data_config=data_config,
        task=task,
        data_dir=eval_config.data_dir,
        model_type = model.embed_config.model,
        distributed=False,
    )

    classifier = MultiLabelMLP(
        embedder=model.embedder,
        task=task,
        num_classes=module.num_classes(),
        config=eval_config,
    )

    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir,
            monitor="valid_auprc",
            mode="min",
            save_top_k=3,
            filename="{epoch}-{valid_auprc:.2f}"
        ),
        # EarlyStopping(
        #     monitor="valid_auprc",
        #     mode="max",
        #     patience=4,
        # ),
    ]
    if eval_config.dev_run:
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
        max_epochs=15,
        precision="bf16-mixed",
        val_check_interval=1.0,
    )
    trainer.fit(
        classifier,
        datamodule=module,
    )
    final_predictions = trainer.predict(
        model=classifier,
        dataloaders=module.val_dataloader(),
        return_predictions=True
    )

    all_logits, all_labels = zip(*final_predictions)
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    output_dir = Path(output_dir)
    torch.save(all_logits, output_dir / "val_logits.pt")
    torch.save(all_labels, output_dir / "val_labels.pt")

    final_fmax = _calc_fmax(all_logits, all_labels)
    final_auprc = multilabel_average_precision(
        all_logits,
        all_labels,
        num_labels=module.num_classes(),
    )
    print(f"Final Fmax: {final_fmax.item():0.3f}")
    print(f"Final AUPRC: {final_auprc.item():0.3f}")

    logger.experiment.finish()
