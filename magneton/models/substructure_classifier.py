import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import lightning as L
import torch
import torch.distributed as dist
import torch.nn as nn
from torch_scatter import scatter_max, scatter_mean
from torchmetrics import Accuracy, F1Score, MetricCollection

from magneton.config import PipelineConfig
from magneton.core_types import SubstructType
from magneton.data.core import Batch, LabeledSubstructure

from .base_models import BaseModelFactory
from .utils import describe_tensor, parse_hidden_dims

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttentionPool(nn.Module):
    """
    Simple attention pooling with a single learnable query vector.

    Input: (L, D) tensor of residue embeddings
    Output: (D,) pooled embedding
    """
    def __init__(self, embed_dim: int, proj_dim: int | None = None):
        super().__init__()
        self.embed_dim = embed_dim
        proj_dim = proj_dim or embed_dim
        # projectors for keys and values
        self.key_proj = nn.Linear(embed_dim, proj_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, proj_dim, bias=False)

        # Initialize query vector to zeros to default to mean pool (i.e.
        # attention weights will be uniform after softmax)
        self.query = nn.Parameter(torch.zeros(proj_dim))
        # small scaling for numerical stability
        self.scale = proj_dim ** 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (L, D)
        returns: (D_proj,) pooled embedding (same dtype/device as x)
        """
        if x.numel() == 0:
            # If empty input, return zeros of appropriate dim on same device/dtype
            return torch.zeros(self.key_proj.out_features, device=x.device, dtype=x.dtype)

        # keys/values: (L, P)
        keys = self.key_proj(x)       # (L, P)
        values = self.value_proj(x)   # (L, P)
        # scores: (L,)
        # use dot-product between keys and query
        scores = keys.matmul(self.query) / self.scale
        attn = torch.softmax(scores, dim=0)  # (L,)
        # weighted sum of values -> (P,)
        pooled = (attn.unsqueeze(-1) * values).sum(dim=0)
        return pooled


class SubstructureClassifier(L.LightningModule):
    """Substructure classification model building on top of base models.

    This is the implementation of substructure classification using mean
    pooling and small per-substructure MLP classifier heads. Some important
    parameters:
        - `config.model.model_params.hidden_dims`: Layer widths for substructure
            classification heads.
        - `config.model.frozen_embedder`: If set to False, this performs
            substructure tuning via full finetuning of the base model.
        - `config.training.loss_strategy`: If set to "ewc", this will first
            compute per-parameter EWC weights by making one pass over the training
            set to compute the weights.
        - `config.training.reuse_ewc_weights`: If set to the path of a model
            previously trained using EWC, this will load the computed EWC weights
            from that model to avoid recomputing here (since the weights are
            just specific to the dataset and base model's original objective, not
            the substructure-tuning objective).

    Args:
        - config (PipelineConfig): Full config for this run.
        - num_classes (int | dict[SubstructType, int]): Number of classes for substructure
            classification heads. If an int, just use one head for all substructures.
            If a dict, the key is the substructure type name and the value is the
            number of classes for that substructure type.
        - load_pretrained_fisher (bool):

    """

    DEFAULT_HEAD_KEY = "default"

    def __init__(
        self,
        config: PipelineConfig,
        num_classes: int | dict[SubstructType, int],
        load_pretrained_fisher: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Normalize num_classes -> dict so runtime logic is uniform
        if isinstance(num_classes, int):
            num_classes = {self.DEFAULT_HEAD_KEY: num_classes}
            self.use_default = True
        else:
            self.use_default = False
        self.num_classes = num_classes

        self.model_config = config.model
        self.train_config = config.training
        self.embed_config = config.base_model

        # Embedder
        self.base_model = BaseModelFactory.create_base_model(
            self.embed_config, frozen=self.model_config.frozen_base_model
        )

        # Build MLP heads (ModuleDict)
        embed_dim = self.base_model.get_embed_dim()
        hidden_dims = parse_hidden_dims(
            raw_dims=self.model_config.model_params["hidden_dims"], embed_dim=embed_dim
        )

        mlps = {}
        for head_name, n_cls in self.num_classes.items():
            layers = []
            prev = embed_dim
            for h in hidden_dims:
                layers.extend(
                    [
                        nn.Linear(prev, h),
                        nn.ReLU(),
                        nn.Dropout(self.model_config.model_params["dropout_rate"]),
                    ]
                )
                prev = h
            layers.append(nn.Linear(prev, n_cls))
            mlps[head_name] = nn.Sequential(*layers)
        self.heads = nn.ModuleDict(mlps)
        logger.info(f"head model: {self.heads}")

        # Pooling method
        self.pooling = config.model.pooling_mechanism
        assert self.pooling in ["mean", "max", "attention"]
        if self.pooling in ["mean", "max"]:
            self.pool_func = self.scatter_pool
        elif self.pooling == "attention":
            self.pool_func = self.attention_pool
            self.att_pool = AttentionPool(embed_dim=embed_dim)
        else:
            raise ValueError(f"unknown pooling type: {self.pooling}")

        # Loss and loss strategy (ewc optional)
        self.loss = nn.CrossEntropyLoss()
        self.loss_strategy = self.train_config.loss_strategy
        if self.loss_strategy == "ewc":
            self.ewc_weight = self.train_config.ewc_weight
            placeholder_vec = torch.zeros_like(
                torch.nn.utils.parameters_to_vector(self.base_model.parameters())
            )
            self.register_buffer("fisher_info", placeholder_vec, persistent=True)

            # Optionally reuse precomputed fisher vector
            if config.training.reuse_ewc_weights is not None:
                ewc_ckpt = torch.load(
                    config.training.reuse_ewc_weights, weights_only=False
                )
                fisher_vec = ewc_ckpt["state_dict"]["fisher_info"]
                self.fisher_info = fisher_vec
                logger.info(
                    f"Loaded precomputed Fisher info complete: {fisher_vec[:10]}"
                )

        # Metrics: per-head Accuracy and F1 for train/val
        self.train_metrics = {}
        self.val_metrics = {}
        for head_name, n_cls in self.num_classes.items():
            head_metrics = {}
            head_metrics[f"{head_name}_accuracy"] = Accuracy(
                task="multiclass", num_classes=n_cls
            )
            head_metrics[f"{head_name}_f1"] = F1Score(
                task="multiclass", num_classes=n_cls
            )

            self.train_metrics[head_name] = MetricCollection(
                head_metrics, prefix="train"
            )
            self.val_metrics[head_name] = self.train_metrics[head_name].clone(
                prefix="valid_"
            )
        self.train_metrics = nn.ModuleDict(self.train_metrics)
        self.val_metrics = nn.ModuleDict(self.val_metrics)

        # Fisher calculation state (for EWC) - can be set externally via `self.calc_fisher_state = True`
        self.calc_fisher_state = False

    # Prediction methods below are used for computing Fisher info if `calc_fisher_state` is set to True,
    # otherwise just behaves as normal prediction returning logits and labels.
    def on_predict_start(self):
        super().on_predict_start()
        if self.calc_fisher_state:
            # Need gradients on embedder to accumulate fisher info
            with torch.inference_mode(False):
                self.base_model._unfreeze(unfreeze_all=True)
                fisher = []
                for p in self.base_model.model.parameters():
                    fisher.append(torch.zeros_like(p, requires_grad=False))
                self.fisher_state = fisher
                self.fisher_samples = 0

    def predict_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0):
        # If we're computing fisher info, compute original-task loss and accumulate squared grads
        if self.calc_fisher_state:
            with torch.inference_mode(False), torch.set_grad_enabled(True):
                orig_loss = self.base_model.calc_original_loss(batch, reduction="sum")
                orig_loss.backward()
                for fisher_params, params in zip(
                    self.fisher_state, self.base_model.model.parameters()
                ):
                    if params.requires_grad and params.grad is not None:
                        fisher_params += params.grad.detach().pow_(2)
                self.fisher_samples += len(batch.protein_ids)
                self.zero_grad()
            return orig_loss  # returned to be reduced/combined by Trainer hooks
        # Otherwise normal predict: return logits dict and labels dict
        logits = self(batch)
        labels = self._gather_labels(batch)
        return logits, labels

    def on_predict_end(self):
        if not self.calc_fisher_state:
            return super().on_predict_end()

        with torch.inference_mode(False):
            fisher_vec_parts = []
            for p in self.fisher_state:
                p.div_(self.fisher_samples)
                fisher_vec_parts.append(torch.flatten(p))
            fisher_vec = torch.cat(fisher_vec_parts)
            logger.info(f"Fisher information calculation complete: {fisher_vec.shape}")

            # Collect Fisher info calculations across ranks if needed.
            if dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
                logger.info(
                    f"Synchronizing Fisher information calculation: rank {rank} / {world_size}"
                )
                describe_tensor(fisher_vec, prefix=f"rank {rank}")

                dist.reduce(fisher_vec, dst=0, op=dist.ReduceOp.SUM)
                dist.barrier()

                if rank == 0:
                    fisher_vec.div_(world_size)

                dist.broadcast(fisher_vec, src=0)
                describe_tensor(fisher_vec, prefix=f"rank {rank}")
                dist.barrier()
            else:
                describe_tensor(fisher_vec, prefix="fisher")

            # register buffer containing fisher vector over embedder params
            self.fisher_info = fisher_vec

        # freeze/unfreeze embedder to original state
        self.base_model._freeze()
        self.base_model._unfreeze(unfreeze_all=False)

        return super().on_predict_end()

    def on_train_start(self):
        # store original embedder params if EWC will be used
        if self.loss_strategy == "ewc":
            self.original_params = torch.nn.utils.parameters_to_vector(
                self.base_model.parameters()
            )
        return super().on_train_start()

    def scatter_pool(
        self,
        protein_embeds: torch.Tensor,
        substructures: list[LabeledSubstructure],
    ) -> dict[str, torch.Tensor]:
        embed_dim = protein_embeds.shape[-1]
        dtype = protein_embeds.dtype

        substruct_embeds = defaultdict(list)
        all_indices = []
        range_ids = []
        for substruct_idx, substructure in enumerate(substructures):
            for start, end in substructure.ranges:
                idx = torch.arange(start, end)
                all_indices.append(idx)
                range_ids.append(torch.full((len(idx),), substruct_idx))
        if len(all_indices) == 0:
            return {}
        all_indices = torch.cat(all_indices).to(self.device)
        range_ids = torch.cat(range_ids).to(self.device)

        # aggregate embeddings per-substructure
        result = torch.zeros(
            len(substructures), embed_dim, dtype=dtype, device=self.device
        )
        if self.pooling == "mean":
            result = scatter_mean(
                index=range_ids[:, None].expand(-1, embed_dim),
                src=protein_embeds[all_indices],
                dim=0,
                out=result,
            )
        elif self.pooling == "max":
            result, _ = scatter_max(
                index=range_ids[:, None].expand(-1, embed_dim),
                src=protein_embeds[all_indices],
                dim=0,
                out=result,
            )
        else:
            raise RuntimeError(f"unexpected pooling type: {self.pooling}")

        for substruct_idx, substructure in enumerate(substructures):
            substruct_embeds[self._map_head_key(substructure.element_type)].append(
                result[substruct_idx]
            )
        return substruct_embeds

    def attention_pool(
        self,
        protein_embeds: torch.Tensor,
        substructures: list[LabeledSubstructure],
    ) -> dict[str, torch.Tensor]:
        """
        For each substructure, gather its residue embeddings and apply the shared
        AttentionPool module. Returns dict head_name -> list of pooled embeddings.
        """
        if len(substructures) == 0:
            return {}

        device = protein_embeds.device
        substruct_embeds = defaultdict(list)

        for substructure in substructures:
            idxs = []
            for start, end in substructure.ranges:
                idxs.append(torch.arange(start, end, device=device))
            if len(idxs) == 0:
                continue
            idxs = torch.cat(idxs)
            if idxs.numel() == 0:
                continue
            # gather residue embeddings for this substructure: (L, D)
            tokens = protein_embeds[idxs]
            # pooled: (P,) where P == proj_dim used inside AttentionPool
            pooled = self.att_pool(tokens)
            # If the pool projects to a different dim than the embed dim, optionally map back.
            # Here we assume proj_dim == embed_dim so pooled has same dim.
            substruct_embeds[self._map_head_key(substructure.element_type)].append(pooled)

        return substruct_embeds


    def calc_substructure_embeds(
        self,
        protein_embeds: torch.Tensor,
        batch: Batch,
    ) -> dict[str, torch.Tensor]:
        """
        Returns a dict mapping head_name -> stacked embeddings for that head.
        For single-task (only 'default' head), all substructures map to 'default'.
        Order of embeddings per head matches the order returned by _gather_labels.
        """
        substruct_embeds = defaultdict(list)
        # iterate proteins
        for i, prot_substructs in enumerate(batch.substructures):
            this_substruct_embeds = self.pool_func(
                protein_embeds=protein_embeds[i],
                substructures=prot_substructs,
            )
            for head_name, embeds_list in this_substruct_embeds.items():
                substruct_embeds[head_name].extend(embeds_list)

        # stack per-head
        ret = {}
        for head_name, embeds in substruct_embeds.items():
            ret[head_name] = torch.stack(embeds)
        return ret

    def _map_head_key(self, key: str) -> str:
        """Possibly map head model key to default, with error checking.

        Convenience function to check if a given key maps to a head model,
        raising a RuntimeError if not. If using a default head model for
        all substructures, just maps to DEFAULT_HEAD_KEY.
        """
        if self.use_default:
            return self.DEFAULT_HEAD_KEY
        else:
            if key not in self.heads:
                raise RuntimeError(
                    f"unexpected substruct type: {key} , "
                    f"expected one of: {list(self.heads.keys())}"
                )
            return key

    def _gather_labels(self, batch: Batch) -> dict[str, torch.Tensor]:
        """
        Produce labels per head, in the same order as calc_substructure_embeds.
        """
        labels = defaultdict(list)
        for prot_substructs in batch.substructures:
            for s in prot_substructs:
                labels[self._map_head_key(s.element_type)].append(int(s.label))

        for k in list(labels.keys()):
            labels[k] = torch.tensor(labels[k], device=self.device)
        return labels

    def forward(self, batch: Batch) -> dict[str, torch.Tensor]:
        """Calculate embeddings for different substructure types."""
        protein_embeds = self.base_model.embed_batch(batch)
        substruct_embeds = self.calc_substructure_embeds(protein_embeds, batch)
        logits = {h: self.heads[h](embs) for h, embs in substruct_embeds.items()}
        return logits

    def training_step(self, batch: Batch, batch_idx: int):
        logits = self(batch)
        labels_per = self._gather_labels(batch)

        losses = []
        for head_name, head_logits in logits.items():
            labels = labels_per[head_name]
            loss = self.loss(head_logits, labels)
            losses.append(loss)

            self.train_metrics[head_name](head_logits, labels)

            self.log(f"train_{head_name}_loss", loss, sync_dist=False)
            self.log(f"{head_name}_eff_batch_size", len(labels), sync_dist=False)
        total_loss = torch.stack(losses).sum()

        # EWC penalty (if configured and fisher_info/original_params exist)
        if self.loss_strategy == "ewc":
            curr_params_vec = torch.nn.utils.parameters_to_vector(
                self.base_model.parameters()
            )
            ewc_loss = (
                (curr_params_vec - self.original_params).pow(2) * self.fisher_info
            ).sum()
            total_loss = total_loss + self.ewc_weight * ewc_loss
            self.log("ewc_loss", ewc_loss, sync_dist=False)

        self.log("train_loss", total_loss, sync_dist=True)

        if batch_idx % 50 == 0:
            for metrics in self.train_metrics.values():
                self.log_dict(metrics, sync_dist=True)
        return total_loss

    @torch.inference_mode()
    def validation_step(self, batch: Batch, batch_idx: int):
        logits = self(batch)
        labels_per = self._gather_labels(batch)

        losses = []
        for head_name, head_logits in logits.items():
            labels = labels_per[head_name]
            loss = self.loss(head_logits, labels)
            losses.append(loss)

            self.val_metrics[head_name].update(head_logits, labels)

            self.log(f"val_{head_name}_loss", loss, sync_dist=False)

        total_loss = torch.stack(losses).sum()
        self.log("val_loss", total_loss, sync_dist=True)

        # Also compute and log original-task loss from embedder
        orig_loss = self.base_model.calc_original_loss(batch)
        self.log("orig_loss", orig_loss, sync_dist=True)

    def on_validation_epoch_end(self):
        for metrics in self.val_metrics.values():
            self.log_dict(metrics, sync_dist=True)
        return super().on_validation_epoch_end()

    def configure_optimizers(self):
        optim_params = [
            {
                "params": self.heads.parameters(),
                "lr": self.train_config.learning_rate,
                "weight_decay": self.train_config.weight_decay,
            }
        ]
        if not self.model_config.frozen_base_model:
            optim_params.append(
                {
                    "params": self.base_model.parameters(),
                    "lr": self.train_config.base_model_learning_rate,
                    "weight_decay": self.train_config.base_model_weight_decay,
                }
            )
        optimizer = torch.optim.AdamW(optim_params)
        return optimizer

    def name(self) -> str:
        return f"{self.embed_config.model}-mlp"

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Modify checkpointing logic to not dump the underlying base model weights if frozen."""
        # If embedder is not frozen, then just use the default state dict with all weights
        if not self.model_config.frozen_base_model:
            return
        # Otherwise overwrite state dict with just the head weights
        checkpoint["state_dict"] = self.heads.state_dict()

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Path,
        map_location: torch.device = None,
        strict: bool = True,
        **kwargs,
    ) -> None:
        """Flexible loading from checkpoint.

        Change here is to not expect full base model weights if the
        base model was frozen during training.
        """
        checkpoint = torch.load(
            checkpoint_path,
            weights_only=False,
            map_location=map_location,
        )
        hparams = checkpoint["hyper_parameters"]
        hparams.update(kwargs)

        model = cls(**hparams)
        if hparams["config"].model.frozen_base_model:
            model.heads.load_state_dict(checkpoint["state_dict"], strict=strict)
        else:
            model.load_state_dict(checkpoint["state_dict"], strict=strict)

        return model
