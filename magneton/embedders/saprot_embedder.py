from dataclasses import dataclass
from typing import Literal, Set

import numpy as np
import torch
from torch.nn import CrossEntropyLoss

from magneton.data.model_specific.saprot import SaProtBatch
from magneton.types import DataType

from .esm_transformers_base import ESMBaseConfig, ESMBaseEmbedder

SAPROT_35M = "35m"
SAPROT_650M = "650m"


@dataclass(kw_only=True)
class SaProtConfig(ESMBaseConfig):
    model_size: Literal[SAPROT_35M, SAPROT_650M] = SAPROT_35M


class SaProtEmbedder(ESMBaseEmbedder):
    def __init__(
        self,
        config: SaProtConfig,
        frozen: bool = True,
    ):
        super().__init__(config, frozen=frozen)
        self.model_size = config.model_size

    def embed_batch(
        self,
        batch: SaProtBatch,
        protein_level: bool = False,
        zero_non_residue_embeds: bool = False,
    ) -> torch.Tensor:
        """Embed a batch of pre-tokenized protein sequences"""
        return self._embed_batch(
            token_tensor=batch.tokenized_sa_seq,
            protein_level=protein_level,
            pooling_method="mean",
            zero_non_residue_embeds=zero_non_residue_embeds,
        )

    def forward_for_contact(
        self,
        batch: SaProtBatch,
    ) -> torch.Tensor:
        return super().forward_for_contact(batch.tokenized_sa_seq)

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
        batch: SaProtBatch,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Calculate SaProt's MLM loss for a batch.

        SaProt's MLM loss is slightly different from normal, instead of doing MLM on the
        final token space (i.e. the structure-aware tokens), they only mask the amino
        acid and leave the structure token information present. Since their tokens are
        the cross product of amino acid tokens and structure tokens, it's easiest to do
        this by just recreating the tokens from the original sequence strings.

        Note: if this seems to be a bottleneck when training EWC models, we can push this
        into the dataloader instead.
        """

        # Perform the amino acid-level masking, keeping track of which indices we've masked
        masked_seqs = []
        masked_row_idxs = []
        masked_col_idxs = []
        for row_idx, (aa_seq, struct_seq) in enumerate(
            zip(batch.seqs, batch.struct_seqs)
        ):
            probs = torch.rand(len(aa_seq), generator=self.rng)
            this_masked_seq = []
            for col_idx, (aa, struct_tok, prob) in enumerate(
                zip(aa_seq, struct_seq, probs)
            ):
                if prob < self.mask_prob:
                    aa = "#"
                    # +1 to col idx for the CLS token that will be added
                    masked_row_idxs.append(row_idx)
                    masked_col_idxs.append(col_idx + 1)
                this_masked_seq.append(f"{aa}{struct_tok}")
            masked_seqs.append("".join(this_masked_seq))

        # Tokenize the masked sequences.
        masked_tokenized_sa_seq = self.tokenizer(
            masked_seqs, return_tensors="pt", padding=True
        )["input_ids"]
        masked_tokenized_sa_seq = masked_tokenized_sa_seq.to(
            device=batch.tokenized_sa_seq.device,
            dtype=batch.tokenized_sa_seq.dtype,
        )

        # Map our masked indices which are (row, col) to flattened indices in the tokenized tensor
        flat_masked_idxs = torch.from_numpy(
            np.ravel_multi_index(
                [masked_row_idxs, masked_col_idxs], masked_tokenized_sa_seq.shape
            )
        ).to(device=masked_tokenized_sa_seq.device)

        # Store original mask values for loss calculation
        orig_values_flat = batch.tokenized_sa_seq.reshape(-1)[flat_masked_idxs]

        logits = self.model.forward(masked_tokenized_sa_seq).logits

        # Flatten logits along batch dim and get logits for just masked positions
        masked_pos_logits = logits.reshape(-1, logits.shape[-1])[flat_masked_idxs]
        return CrossEntropyLoss(reduction=reduction)(
            masked_pos_logits, orig_values_flat
        )

    def model_name(self) -> str:
        return f"SaProt-{self.model_size}"

    @classmethod
    def get_required_input_type(cls) -> Set[DataType]:
        return {DataType.SEQ, DataType.STRUCT}
