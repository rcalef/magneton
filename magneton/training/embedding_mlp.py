from collections import defaultdict
from typing import Dict

import lightning as L
import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score, MetricCollection
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
        self.num_classes = num_classes

        self.embedder = EmbedderFactory.create_embedder(
            self.embed_config,
            frozen=self.model_config.frozen_embedder,
        )

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build MLP layers
        layers = []
        prev_dim = self.embedder.get_embed_dim()
        for hidden_dim in self.model_config.model_params["hidden_dims"]:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.model_config.model_params["dropout_rate"]),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers)

        self.loss = nn.CrossEntropyLoss()

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.f1 = F1Score(task="multiclass", num_classes=num_classes)

    def configure_model(self):
        print(f"MLP Device: {self.device}")

    def calc_substructure_embeds(
        self,
        # num_proteins X max_length X embed_dim
        protein_embeds: torch.Tensor,
        batch: SubstructureBatch,
    ) -> torch.Tensor:
        # print(protein_embeds.shape)
        embed_dim = protein_embeds.shape[-1]
        dtype = protein_embeds.dtype

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

            all_indices = torch.cat(all_indices).to(self.device)
            range_ids = torch.cat(range_ids).to(self.device)

            # num_substructs[i] X embed_dim
            result = torch.zeros(
                len(prot_substructs),
                embed_dim,
                dtype=dtype,
                device=self.device,
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
        return substruct_embeds

    def forward(
        self,
        batch: SubstructureBatch,
    ) -> torch.Tensor:
        # num_substructs X embed_dim
        protein_embeds = self.embedder.embed_batch(batch)
        substruct_embeds = self.calc_substructure_embeds(protein_embeds, batch)

        # num_substructs X num_classes
        return self.model(substruct_embeds)

    def training_step(self, batch: SubstructureBatch, batch_idx) -> torch.Tensor:
        logits = self(batch)
        labels = torch.tensor(
            [
                substruct.label
                for prot_substructs in batch.substructures
                for substruct in prot_substructs
            ],
            device=logits.device,
        )

        loss = self.loss(logits, labels)
        preds = torch.argmax(logits, dim=1)

        acc = self.train_acc(preds, labels)
        # Revisit this if it seems slower
        self.log("train_loss", loss, sync_dist=True)
        self.log("train_acc", acc, sync_dist=True)
        self.log("eff_batch_size", batch.total_length(), sync_dist=True)
        return loss

    def validation_step(self, batch: SubstructureBatch, batch_idx):
        logits = self(batch)
        labels = torch.tensor(
            [
                substruct.label
                for prot_substructs in batch.substructures
                for substruct in prot_substructs
            ],
            device=logits.device,
        )

        loss = self.loss(logits, labels)
        preds = torch.argmax(logits, dim=1)

        acc = self.val_acc(preds, labels)
        f1 = self.f1(preds, labels)

        # Revisit this if it seems slower
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_acc", acc, sync_dist=True)
        self.log("val_f1", f1, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
        )
        return optimizer

    def name(self) -> str:
        return f"{self.embed_config.model}-mlp"


class MultitaskEmbeddingMLP(L.LightningModule):
    def __init__(
        self,
        config: PipelineConfig,
        num_classes: Dict[str, int],
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_config = config.model
        self.train_config = config.training
        self.embed_config = config.embedding
        self.num_classes = num_classes

        self.embedder = EmbedderFactory.create_embedder(
            self.embed_config,
            frozen=self.model_config.frozen_embedder,
        )

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build MLP layers and metric loggers
        substruct_types = config.data.interpro_types
        mlps = {}
        for substruct_type in substruct_types:
            num_classes = self.num_classes[substruct_type]
            layers = []
            prev_dim = self.embedder.get_embed_dim()
            for hidden_dim in self.model_config.model_params["hidden_dims"]:
                layers.extend(
                    [
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(self.model_config.model_params["dropout_rate"]),
                    ]
                )
                prev_dim = hidden_dim

            layers.append(nn.Linear(prev_dim, num_classes))
            mlps[substruct_type] = nn.Sequential(*layers)

        self.mlps = nn.ModuleDict(mlps)
        self.loss = nn.CrossEntropyLoss()

    def configure_model(self):
        print(f"MLP Device: {self.device}")

    def calc_substructure_embeds(
        self,
        # num_proteins X max_length X embed_dim
        protein_embeds: torch.Tensor,
        batch: SubstructureBatch,
    ) -> Dict[str, torch.Tensor]:

        embed_dim = protein_embeds.shape[-1]
        dtype = protein_embeds.dtype

        substruct_embeds = defaultdict(list)
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

            all_indices = torch.cat(all_indices).to(self.device)
            range_ids = torch.cat(range_ids).to(self.device)

            # num_substructs[i] X embed_dim
            result = torch.zeros(
                len(prot_substructs),
                embed_dim,
                dtype=dtype,
                device=self.device,
            )
            result = scatter_mean(
                index=range_ids[:, None].expand(-1, embed_dim),
                src=protein_embeds[i][all_indices],
                dim=0,
                out=result,
            )
            for substruct_idx, substructure in enumerate(prot_substructs):
                substruct_embeds[substructure.element_type].append(
                    result[substruct_idx]
                )
        # num_substructs[substruct_type] X embed_dim
        ret = {}
        for substruct_type, embeds in substruct_embeds.items():
            ret[substruct_type] = torch.stack(embeds)
        return ret

    def forward(self, batch: SubstructureBatch) -> Dict[str, torch.Tensor]:
        protein_embeds = self.embedder.embed_batch(batch)

        # num_substructs X embed_dim
        substruct_embeds = self.calc_substructure_embeds(protein_embeds, batch)

        # num_substructs X num_classes
        logits = {}
        for substruct_type, embeds in substruct_embeds.items():
            logits[substruct_type] = self.mlps[substruct_type](embeds)
        return logits

    def training_step(self, batch: SubstructureBatch, batch_idx) -> torch.Tensor:
        logits = self(batch)

        for substruct_type, substruct_logits in logits.items():
            dtype = substruct_logits.dtype
            break

        total_loss = torch.tensor(0, device=self.device, dtype=dtype)
        for substruct_type, substruct_logits in logits.items():

            labels = torch.tensor(
                [
                    substruct.label
                    for prot_substructs in batch.substructures
                    for substruct in prot_substructs
                    if substruct.element_type == substruct_type
                ],
                device=self.device,
            )

            loss = self.loss(substruct_logits, labels)
            total_loss += loss

            with torch.no_grad():
                preds = torch.argmax(substruct_logits, dim=1)
                acc = (preds == labels).float().mean()

            # Revisit this if it seems slower
            self.log(f"{substruct_type}_train_loss", loss, sync_dist=True)
            self.log(f"{substruct_type}_train_acc", acc, sync_dist=True)
            self.log(f"{substruct_type}_eff_batch_size", len(labels), sync_dist=True)

        self.log("train_loss", total_loss, sync_dist=True)
        return loss

    def validation_step(self, batch: SubstructureBatch, batch_idx):
        logits = self(batch)

        for substruct_type, substruct_logits in logits.items():
            dtype = substruct_logits.dtype
            break

        total_loss = torch.tensor(0, device=self.device, dtype=dtype)
        for substruct_type, substruct_logits in logits.items():

            labels = torch.tensor(
                [
                    substruct.label
                    for prot_substructs in batch.substructures
                    for substruct in prot_substructs
                    if substruct.element_type == substruct_type
                ],
                device=self.device,
            )

            loss = self.loss(substruct_logits, labels)
            total_loss += loss

            with torch.no_grad():
                preds = torch.argmax(substruct_logits, dim=1)
                acc = (preds == labels).float().mean()

            # Revisit this if it seems slower
            self.log(f"{substruct_type}_val_loss", loss, sync_dist=True)
            self.log(f"{substruct_type}_val_acc", acc, sync_dist=True)
            self.log(f"{substruct_type}_eff_batch_size", len(labels), sync_dist=True)

        self.log("val_loss", total_loss, sync_dist=True)

        # Calculate loss for original task
        orig_loss = self.embedder.calc_original_loss(batch)
        self.log("orig_loss", orig_loss, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
        )
        return optimizer

    def name(self) -> str:
        return f"{self.embed_config.model}-mlp"
