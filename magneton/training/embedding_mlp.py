from collections import defaultdict
from typing import Dict, Tuple

import lightning as L
import torch
import torch.distributed as dist
import torch.nn as nn
from torchmetrics import Accuracy, F1Score
from torch_scatter import scatter_mean

from magneton.config import PipelineConfig
from magneton.data.core import Batch
from magneton.embedders.factory import EmbedderFactory
from magneton.utils import describe_tensor

# Some older runs used different namings, use this function as a catch-all for compatibility
# fixes
def compatibility_fixes(
    cfg: PipelineConfig,
):
    if cfg.embedding.model == "esmc_300m":
        cfg.embedding.model = "esmc"


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


class EmbeddingMLP(L.LightningModule):
    def __init__(
        self,
        config: PipelineConfig,
        num_classes: int,
        load_pretrained_fisher: bool = False,
        for_contact_prediction: bool = False,
    ):
        super().__init__()
        compatibility_fixes(config)
        self.save_hyperparameters()
        self.model_config = config.model
        self.train_config = config.training
        self.embed_config = config.embedding
        self.num_classes = num_classes

        self.embed_config.model_params["for_contact_prediction"] = for_contact_prediction
        self.embedder = EmbedderFactory.create_embedder(
            self.embed_config,
            frozen=self.model_config.frozen_embedder,
        )

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build MLP layers
        layers = []
        embed_dim = self.embedder.get_embed_dim()
        hidden_dims = parse_hidden_dims(
            raw_dims=self.model_config.model_params["hidden_dims"],
            embed_dim=embed_dim
        )
        prev_dim = embed_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.model_config.model_params["dropout_rate"]),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

        self.loss = nn.CrossEntropyLoss()
        self.loss_strategy = self.train_config.loss_strategy
        if self.loss_strategy == "ewc":
            self.ewc_weight = self.train_config.ewc_weight
            if config.training.reuse_ewc_weights is not None:
                ewc_ckpt = torch.load(config.training.reuse_ewc_weights, weights_only=False)
                fisher_vec = ewc_ckpt["state_dict"]["fisher_info"]
                self.register_buffer(
                    "fisher_info",
                    fisher_vec,
                    persistent=True,
                )
                print(f"Loaded precomputed Fisher info complete: {fisher_vec[:10]}")

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.f1 = F1Score(task="multiclass", num_classes=num_classes)

        self.calc_fisher_state = False
        if load_pretrained_fisher:
            placeholder_vec = torch.zeros_like(torch.nn.utils.parameters_to_vector(self.embedder.parameters()))
            self.register_buffer(
                "fisher_info",
                placeholder_vec,
                persistent=True,
            )

    def on_predict_start(self):
        super().on_predict_start()
        # Need to track gradients if we're calculating Fisher information
        # for EWC
        if self.calc_fisher_state:
            with torch.inference_mode(False):
                self.embedder._unfreeze(unfreeze_all=True)
                fisher = []
                for params in self.embedder.model.parameters():
                    fisher.append(torch.zeros_like(
                        params,
                        requires_grad=False,
                    ))
                self.fisher_state = fisher
                self.fisher_samples = 0

    def predict_step(
        self,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        if self.calc_fisher_state:
            with torch.inference_mode(False), torch.set_grad_enabled(True):
                orig_loss = self.embedder.calc_original_loss(batch, reduction="sum")
                orig_loss.backward()

                for (fisher_params, params) in zip(
                    self.fisher_state,
                    self.embedder.model.parameters()
                ):
                    if params.requires_grad:
                        fisher_params += params.grad.detach().pow_(2)

                self.fisher_samples += len(batch.protein_ids)
                self.zero_grad()
            return orig_loss
        else:
            loss, logits, labels = self._calc_substruct_outputs(batch)
            return loss, logits, labels

    def on_predict_end(self):
        # Nothing to do here if not calculating Fisher
        # information
        if not self.calc_fisher_state:
            return super().on_predict_end()

        # Otherwise, a lot to do.
        with torch.inference_mode(False):
            fisher_vec = []
            for fisher_params in self.fisher_state:
                fisher_params.div_(self.fisher_samples)
                fisher_vec.append(torch.flatten(fisher_params))
            fisher_vec = torch.cat(fisher_vec)

            print(f"Fisher information calculation complete: {fisher_vec.shape}")

            # If operating in distributed setting, need to
            # handle accordingly
            if dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
                print(f"Synchronizing Fisher information calculation: rank {rank} / {world_size}")
                describe_tensor(fisher_vec, prefix=f"rank {rank}")

                dist.reduce(fisher_vec, dst=0, op=dist.ReduceOp.SUM)
                print(f"rank {rank}: reducing Fisher information calculation")
                dist.barrier()

                if rank == 0:
                    fisher_vec.div_(world_size)

                dist.broadcast(fisher_vec, src=0)
                print(f"rank {rank}: reduced Fisher information calculation")
                describe_tensor(fisher_vec, prefix=f"rank {rank}")

                dist.barrier()
            else:
                describe_tensor(fisher_vec, prefix=f"fisher")

            self.register_buffer(
                "fisher_info",
                fisher_vec,
                persistent=True,
            )

        self.embedder._freeze()
        self.embedder._unfreeze(unfreeze_all=False)

        return super().on_predict_end()

    def on_train_start(self):
        if self.loss_strategy == "ewc":
            # Store original parameters
            self.original_params = torch.nn.utils.parameters_to_vector(
                self.embedder.parameters()
            )

        return super().on_train_start()

    def calc_substructure_embeds(
        self,
        # num_proteins X max_length X embed_dim
        protein_embeds: torch.Tensor,
        batch: Batch,
    ) -> torch.Tensor:
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
        batch: Batch,
    ) -> torch.Tensor:
        # num_substructs X embed_dim
        protein_embeds = self.embedder.embed_batch(batch)
        substruct_embeds = self.calc_substructure_embeds(protein_embeds, batch)

        # num_substructs X num_classes
        return self.mlp(substruct_embeds)

    def _calc_substruct_outputs(
            self,
            batch: Batch,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return loss, logits, labels


    def training_step(self, batch: Batch, batch_idx) -> torch.Tensor:
        loss, logits, labels = self._calc_substruct_outputs(batch)

        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, labels)

        if self.loss_strategy == "ewc":
            curr_params_vec = torch.nn.utils.parameters_to_vector(
                self.embedder.parameters(),
            )

            self.log("substruct_loss", loss, sync_dist=True)
            ewc_loss = ((curr_params_vec - self.original_params).pow_(2) * self.fisher_info).sum()
            loss += self.ewc_weight * ewc_loss
            self.log("ewc_loss", ewc_loss, sync_dist=True)

        # Revisit this if it seems slower
        self.log("train_loss", loss, sync_dist=True)
        self.log("train_acc", acc, sync_dist=True)
        self.log("eff_batch_size", batch.total_length(), sync_dist=True)
        return loss

    def validation_step(self, batch: Batch, batch_idx):
        loss, logits, labels = self._calc_substruct_outputs(batch)

        preds = torch.argmax(logits, dim=1)

        acc = self.val_acc(preds, labels)
        f1 = self.f1(preds, labels)

        # Revisit this if it seems slower
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_acc", acc, sync_dist=True)
        self.log("val_f1", f1, sync_dist=True)

        # Calculate loss for original task
        orig_loss = self.embedder.calc_original_loss(batch)
        self.log("orig_loss", orig_loss, on_step=True, sync_dist=True)

    def configure_optimizers(self):
        optim_params = []
        # MLP params
        optim_params.append({
            "params": self.mlp.parameters(),
            "lr": self.train_config.learning_rate,
            "weight_decay": self.train_config.weight_decay,
        })
        if not self.model_config.frozen_embedder:
            optim_params.append({
                "params": self.embedder.parameters(),
                "lr": self.train_config.embedding_learning_rate,
                "weight_decay": self.train_config.embedding_weight_decay,
            })
        optimizer = torch.optim.AdamW(
            optim_params,
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

        # Build MLP layers and metric loggers
        embed_dim = self.embedder.get_embed_dim()
        hidden_dims = parse_hidden_dims(
            raw_dims=self.model_config.model_params["hidden_dims"],
            embed_dim=embed_dim
        )

        substruct_types = config.data.substruct_types
        mlps = {}
        for substruct_type in substruct_types:
            num_classes = self.num_classes[substruct_type]
            layers = []

            prev_dim = embed_dim
            for hidden_dim in hidden_dims:
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
            print(f"substruct MLP: {substruct_type}, num classes={num_classes}")

        self.mlps = nn.ModuleDict(mlps)
        self.loss = nn.CrossEntropyLoss()

    def configure_model(self):
        print(f"MLP Device: {self.device}")

    def calc_substructure_embeds(
        self,
        # num_proteins X max_length X embed_dim
        protein_embeds: torch.Tensor,
        batch: Batch,
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

    def forward(self, batch: Batch) -> Dict[str, torch.Tensor]:
        protein_embeds = self.embedder.embed_batch(batch)

        # num_substructs X embed_dim
        substruct_embeds = self.calc_substructure_embeds(protein_embeds, batch)

        # num_substructs X num_classes
        logits = {}
        for substruct_type, embeds in substruct_embeds.items():
            logits[substruct_type] = self.mlps[substruct_type](embeds)
        return logits

    def training_step(self, batch: Batch, batch_idx) -> torch.Tensor:
        logits = self(batch)

        for substruct_type, substruct_logits in logits.items():
            dtype = substruct_logits.dtype
            break

        losses = []
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
            losses.append(loss)

            with torch.no_grad():
                preds = torch.argmax(substruct_logits, dim=1)
                acc = (preds == labels).float().mean()

            # Revisit this if it seems slower
            self.log(f"{substruct_type}_train_loss", loss, sync_dist=True)
            self.log(f"{substruct_type}_train_acc", acc, sync_dist=True)
            self.log(f"{substruct_type}_eff_batch_size", len(labels), sync_dist=True)

        total_loss = torch.stack(losses).sum()
        self.log("train_loss", total_loss, sync_dist=True)
        return total_loss

    @torch.inference_mode()
    def validation_step(self, batch: Batch, batch_idx):
        logits = self(batch)

        for substruct_type, substruct_logits in logits.items():
            dtype = substruct_logits.dtype
            break

        losses = []
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
            losses.append(loss)

            preds = torch.argmax(substruct_logits, dim=1)
            acc = (preds == labels).float().mean()

            # Revisit this if it seems slower
            self.log(f"{substruct_type}_val_loss", loss, sync_dist=True)
            self.log(f"{substruct_type}_val_acc", acc, sync_dist=True)
            self.log(f"{substruct_type}_eff_batch_size", len(labels), sync_dist=True)

        total_loss = torch.stack(losses).sum()
        self.log("val_loss", total_loss, sync_dist=True)

        # Calculate loss for original task
        orig_loss = self.embedder.calc_original_loss(batch)
        self.log("orig_loss", orig_loss, sync_dist=True)

    def configure_optimizers(self):
        optim_params = []
        # MLP params
        optim_params.append({
            "params": self.mlps.parameters(),
            "lr": self.train_config.learning_rate,
            "weight_decay": self.train_config.weight_decay,
        })
        if not self.model_config.frozen_embedder:
            optim_params.append({
                "params": self.embedder.parameters(),
                "lr": self.train_config.embedding_learning_rate,
                "weight_decay": self.train_config.embedding_weight_decay,
            })
        optimizer = torch.optim.AdamW(
            optim_params,
        )
        return optimizer

    def name(self) -> str:
        return f"{self.embed_config.model}-mlp"
