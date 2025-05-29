import torch
import torch.nn as nn
import lightning as L

from esm.tokenization import get_esmc_model_tokenizers
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torchmetrics import (
    Accuracy,
    AveragePrecision,
    MetricCollection,
)

from magneton.config import EvalConfig
from magneton.data.supervised_dataset import (
    SupervisedBatch,
    DeepFRIDataConfig,
    DeepFRIDataModule,
)
from magneton.embedders.base_embedder import BaseEmbedder
from magneton.training.embedding_mlp import EmbeddingMLP, MultitaskEmbeddingMLP

class MultiLabelMLP(L.LightningModule):
    def __init__(
        self,
        embedder: BaseEmbedder,
        task: str,
        num_classes: int,
        pad_id: int,
        eos_id: int,
        config: EvalConfig,
    ):
        super().__init__()
        self.config = config
        self.task = task
        self.num_classes = num_classes
        self.embedder = embedder
        self.pad_id = pad_id
        self.eos_id = eos_id

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

    def forward(
        self,
        batch: SupervisedBatch,
    ) -> torch.Tensor:
        # batch_size (num_proteins) X max_len X embed_dim
        protein_embeds = self.embedder.embed_batch(batch)

        attention_mask = (
            (batch.tokenized_seq != self.pad_id) &
            (batch.tokenized_seq != self.eos_id)
        )[:, 1:].unsqueeze(-1)

        # Zero out the embeddings of pad tokens
        masked_embeddings = protein_embeds * attention_mask

        # Sum the embeddings along the sequence length dimension
        summed_embeddings = torch.sum(masked_embeddings, dim=1)

        # Get the actual sequence lengths (number of non-pad tokens)
        seq_lengths = attention_mask.sum(dim=1, dtype=protein_embeds.dtype)

        # Divide the summed embeddings by the sequence lengths
        pooled_embeddings = summed_embeddings / seq_lengths

        # proteins X num_classes
        return self.mlp(pooled_embeddings)

    def training_step(self, batch: SupervisedBatch, batch_idx: int) -> torch.Tensor:
        logits = self(batch)

        labels = batch.labels.to(dtype=logits.dtype)
        loss = self.loss(logits, labels)

        # Revisit this if it seems slower
        self.log("train_loss", loss, sync_dist=True)
        if batch_idx % 50 == 0:
            self.train_metrics(logits, batch.labels)
            self.log_dict(self.train_metrics)

        return loss

    def validation_step(self, batch: SupervisedBatch, batch_idx):
        logits = self(batch)

        # preds = torch.argmax(logits, dim=1)
        # acc = self.train_acc(preds, labels)
        labels = batch.labels.to(dtype=logits.dtype)
        loss = self.loss(logits, labels)

        # Revisit this if it seems slower
        self.log("val_loss", loss, sync_dist=True)
        self.val_metrics(logits, batch.labels)
        self.log_dict(self.val_metrics)

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
    config: EvalConfig,
):
    embedder_type = model.embed_config.model
    if embedder_type in ["esmc", "esmc_300m"]:
        tokenizer = get_esmc_model_tokenizers()
    else:
        tokenizer = None

    if task in ["GO:BP", "GO:CC", "GO:MF"]:
        go_term = task.split(":")[1]
        go_config = DeepFRIDataConfig(
            data_dir=config.data_dir,
            go_term=go_term,
            batch_size=config.model_params["batch_size"],
        )
        module = DeepFRIDataModule(
            config=go_config,
            seq_tokenizer=tokenizer,
        )
    else:
        raise ValueError(f"unknown supervised classification dataset: {task}")

    classifier = MultiLabelMLP(
        embedder=model.embedder,
        task=task,
        num_classes=module.num_classes,
        config=config,
        pad_id=tokenizer.pad_token_id,
        eos_id=tokenizer.eos_token_id,
    )

    # Set up callbacks
    callbacks = [
        # ModelCheckpoint(
        #     dirpath=self.save_dir / f"checkpoints_{self.run_id}",
        #     monitor="val_loss",
        #     mode="min",
        #     save_top_k=3,
        #     filename="{epoch}-{val_loss:.2f}"
        # ),
        EarlyStopping(
            monitor="valid_auprc",
            mode="max",
            patience=4,
        ),
    ]
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
        max_epochs=10,
        precision="bf16-mixed",
        val_check_interval=0.25,
    )
    trainer.fit(
        classifier,
        datamodule=module,
    )
    logger.experiment.finish()
