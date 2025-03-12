from typing import Dict, Any

import lightning as L
import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score
from torch_scatter import scatter_mean

from magneton.config import PipelineConfig
from magneton.embedders.esmc_embedder import SubstructureBatch
from magneton.embedders.factory import EmbedderFactory

class EmbeddingMLP(L.LightningModule):
    def __init__(
        self,
        config: PipelineConfig,
        num_classes: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_config = config.model
        self.train_config = config.training
        self.embed_config = config.embedding

        self.embedder = EmbedderFactory.create_embedder(self.embed_config)

        # Build MLP layers
        layers = []
        prev_dim = self.embedder.get_embed_dim()
        for hidden_dim in self.model_config.model_params["hidden_dims"]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.model_config.model_params["dropout_rate"])
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers)

        self.loss = nn.CrossEntropyLoss()

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.f1 = F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, batch: SubstructureBatch):
        # num_proteins X max_length X embed_dim
        protein_embeds = self.embedder.embed_batch(batch)
        # print(protein_embeds.shape)
        embed_dim = protein_embeds.shape[-1]
        dtype = protein_embeds.dtype
        device = protein_embeds.device

        substruct_embeds = []
        # Iter over proteins
        for i, prot_substructs in enumerate(batch.substructures):
            all_indices = []
            range_ids = []
            # Iter over substructures within protein
            for substruct_idx, substructure in enumerate(prot_substructs):
                # Iter over constituent ranges within substructure
                for start, end in substructure.ranges:
                    idx = torch.arange(start, end)
                    all_indices.append(idx)
                    range_ids.append(torch.full((len(idx),), substruct_idx))

            all_indices = torch.cat(all_indices).to(device)
            range_ids = torch.cat(range_ids).to(device)

            # num_substructs[i] X embed_dim
            result = torch.zeros(
                len(prot_substructs),
                embed_dim,
                dtype=dtype,
                device=device,
            )
            result = scatter_mean(
                index=range_ids[:, None].expand(-1, embed_dim),
                src=protein_embeds[i][all_indices],
                dim=0,
                out=result,
            )
            substruct_embeds.append(result)
        # num_substructs X embed_dim
        substruct_embeds = torch.cat(substruct_embeds)

        # num_substructs X num_classes
        return self.model(substruct_embeds)

    def training_step(self, batch: SubstructureBatch, batch_idx) -> torch.Tensor:
        logits = self(batch)
        labels = torch.tensor([substruct.label for prot_substructs in batch.substructures for substruct in prot_substructs], device=logits.device)

        loss = self.loss(logits, labels)
        preds = torch.argmax(logits, dim=1)

        acc = self.train_acc(preds, labels)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        self.log("eff_batch_size", batch.total_length())
        return loss

    def validation_step(self, batch: SubstructureBatch, batch_idx):
        logits = self(batch)
        labels = torch.tensor([substruct.label for prot_substructs in batch.substructures for substruct in prot_substructs], device=logits.device)

        loss = self.loss(logits, labels)
        preds = torch.argmax(logits, dim=1)

        acc = self.val_acc(preds, labels)
        f1 = self.f1(preds, labels)

        self.log("val_loss", loss)
        self.log("val_acc", acc)
        self.log("val_f1", f1)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
        )
        return optimizer

    def name(self) -> str:
        return f"{self.embed_config.model}-mlp"