from dataclasses import dataclass
from typing import Literal, Set

import torch
from torch.nn import CrossEntropyLoss

from magneton.data.model_specific.esm2 import ESM2Batch
from magneton.types import DataType

from .esm_transformers_base import ESMBaseEmbedder, ESMBaseConfig
from .utils import get_seq_mask

ESM2_150M = "150m"
ESM2_600M = "600m"
ESM2_3B = "3b"

@dataclass(kw_only=True)
class ESM2Config(ESMBaseConfig):
    model_size: Literal[ESM2_150M, ESM2_600M, ESM2_3B] = ESM2_150M


class ESM2Embedder(ESMBaseEmbedder):
    """ESM protein embedding model"""

    def __init__(
        self,
        config: ESM2Config,
        frozen: bool = True,
    ):
        super().__init__(config, frozen=frozen)
        self.model_size = config.model_size

    def embed_batch(
        self,
        batch: ESM2Batch,
        protein_level: bool = False,
        zero_non_residue_embeds: bool = False,
    ) -> torch.Tensor:
        """Embed a batch of pre-tokenized protein sequences"""
        return self._embed_batch(
            token_tensor=batch.tokenized_seq,
            protein_level=protein_level,
            pooling_method="cls",
            zero_non_residue_embeds=zero_non_residue_embeds,
        )

    # the following two functions are deprecated for the current data module setup
    @torch.no_grad()
    def embed_single_protein(self, seq: str) -> torch.Tensor:
        """Process a single protein sequence through ESM"""
        pass

    @torch.no_grad()
    def embed_sequences(self, sequences: list[str]) -> list[torch.Tensor]:
        """Embed multiple protein sequences"""
        pass

    def calc_original_loss(
        self,
        batch: ESM2Batch,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """NOTE: this modifies the original tokenized seq tensor, call last."""
        seqs = batch.tokenized_seq

        ignore_tokens = torch.tensor([
            self.tokenizer.pad_token_id,
            self.tokenizer.cls_token_id,
            self.tokenizer.eos_token_id,
        ], device=seqs.device)

        mask = get_seq_mask(
            seqs,
            ignore_tokens=ignore_tokens,
            rng=self.rng,
            mask_prob=self.mask_prob,
        )
        masked_idxs = torch.where(mask.reshape(-1))[0]

        # Store original mask values for loss calculation
        orig_values_flat = seqs.reshape(-1)[masked_idxs]

        seqs = seqs.masked_fill(mask, self.tokenizer.mask_token_id)

        logits = self.model.forward(seqs).logits

        # Flatten logits along batch dim and get logits for just masked positions
        masked_pos_logits = logits.reshape(-1, logits.shape[-1])[masked_idxs]
        return CrossEntropyLoss(reduction=reduction)(masked_pos_logits, orig_values_flat)

    def model_name(self) -> str:
        return f"SaProt-{self.model_size}"

    @classmethod
    def get_required_input_type(cls) -> Set[DataType]:
        return {DataType.SEQ, DataType.STRUCT}
