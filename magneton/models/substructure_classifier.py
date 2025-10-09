import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import lightning as L
import torch
import torch.distributed as dist
import torch.nn as nn
from torch_scatter import scatter_mean
from torchmetrics import Accuracy, F1Score, MetricCollection

from magneton.config import PipelineConfig
from magneton.core_types import SubstructType
from magneton.data.core import Batch

from .base_models import BaseModelFactory
from .utils import describe_tensor, parse_hidden_dims

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        self.embed_config = config.embedding

        # Embedder
        self.base_model = BaseModelFactory.create_base_model(
            self.embed_config, frozen=self.model_config.frozen_embedder
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

        # Default head name (useful if some batch element types don't match head names
        # e.g. single-task 'default' head)
        self.default_head = next(iter(self.heads.keys()))

        # Loss and loss strategy (ewc optional)
        self.loss = nn.CrossEntropyLoss()
        self.loss_strategy = self.train_config.loss_strategy
        if self.loss_strategy == "ewc":
            self.ewc_weight = self.train_config.ewc_weight
            # Optionally reuse precomputed fisher vector
            if config.training.reuse_ewc_weights is not None:
                ewc_ckpt = torch.load(
                    config.training.reuse_ewc_weights, weights_only=False
                )
                fisher_vec = ewc_ckpt["state_dict"]["fisher_info"]
                self.register_buffer("fisher_info", fisher_vec, persistent=True)
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
        if load_pretrained_fisher:
            placeholder_vec = torch.zeros_like(
                torch.nn.utils.parameters_to_vector(self.base_model.parameters())
            )
            self.register_buffer("fisher_info", placeholder_vec, persistent=True)

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
            self.register_buffer("fisher_info", fisher_vec, persistent=True)

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
        embed_dim = protein_embeds.shape[-1]
        dtype = protein_embeds.dtype

        substruct_embeds = defaultdict(list)
        # iterate proteins
        for i, prot_substructs in enumerate(batch.substructures):
            all_indices = []
            range_ids = []
            for substruct_idx, substructure in enumerate(prot_substructs):
                for start, end in substructure.ranges:
                    idx = torch.arange(start, end)
                    all_indices.append(idx)
                    range_ids.append(torch.full((len(idx),), substruct_idx))
            if len(all_indices) == 0:
                continue
            all_indices = torch.cat(all_indices).to(self.device)
            range_ids = torch.cat(range_ids).to(self.device)

            # aggregate embeddings per-substructure
            result = torch.zeros(
                len(prot_substructs), embed_dim, dtype=dtype, device=self.device
            )
            result = scatter_mean(
                index=range_ids[:, None].expand(-1, embed_dim),
                src=protein_embeds[i][all_indices],
                dim=0,
                out=result,
            )

            for substruct_idx, substructure in enumerate(prot_substructs):
                substruct_embeds[self._map_head_key(substructure.element_type)].append(
                    result[substruct_idx]
                )

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
        if not self.model_config.frozen_embedder:
            optim_params.append(
                {
                    "params": self.base_model.parameters(),
                    "lr": self.train_config.embedding_learning_rate,
                    "weight_decay": self.train_config.embedding_weight_decay,
                }
            )
        optimizer = torch.optim.AdamW(optim_params)
        return optimizer

    def name(self) -> str:
        return f"{self.embed_config.model}-mlp"

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Modify checkpointing logic to not dump the underlying base model weights if frozen."""
        # If embedder is not frozen, then just use the default state dict with all weights
        if not self.model_config.frozen_embedder:
            return
        # Otherwise overwrite state dict with just the head weights
        checkpoint["state_dict"] = self.heads.state_dict()

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Path,
        **kwargs,
    ) -> None:
        """Flexible loading from checkpoint.

        Main changes here are not expecting full base model weights if the
        base model was frozen during training, and separately, handling the presence
        of EWC weights even if this specific config isn't expecting them. The
        converse of expecting EWC weights when they're not present will throw a
        RuntimeError.
        """
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        hparams = checkpoint["hyper_parameters"]
        hparams.update(kwargs)

        model = cls(**hparams)
        if hparams["config"].model.frozen_embedder:
            model.heads.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint["state_dict"])

        return model

