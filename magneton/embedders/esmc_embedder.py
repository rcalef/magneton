import os

from dataclasses import dataclass
from typing import Literal, Set

import torch

from esm.models.esmc import ESMC
from esm.sdk.api import (
    ESMProtein,
)
from esm.tokenization import get_esmc_model_tokenizers
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from magneton.data.model_specific.esmc import ESMCBatch
from magneton.types import DataType
from magneton.utils import get_chunk_idxs

from .base_embedder import BaseConfig, BaseEmbedder
from .utils import get_seq_mask

ESMC_300M = "300m"
ESMC_600M = "600m"

@dataclass(kw_only=True)
class ESMCConfig(BaseConfig):
    weights_path: str
    model_size: Literal[ESMC_300M, ESMC_600M] = ESMC_600M
    use_flash_attn: bool = False
    rep_layer: int = 35
    max_seq_length: int = 2048
    mask_prob: float = 0.15


class ESMCEmbedder(BaseEmbedder):
    def __init__(
        self,
        config: ESMCConfig,
        frozen: bool = True,
    ):
        super().__init__(config)

        self.max_len = config.max_seq_length
        self.rep_layer = config.rep_layer
        self.model_size = config.model_size

        if config.model_size == ESMC_600M:
            self.model = ESMC(
                d_model=1152,
                n_heads=18,
                n_layers=36,
                tokenizer=get_esmc_model_tokenizers(),
                use_flash_attn=config.use_flash_attn,
            )
            self.embed_dim=1152
        elif config.model_size == ESMC_300M:
            self.model = ESMC(
                d_model=960,
                n_heads=15,
                n_layers=30,
                tokenizer=get_esmc_model_tokenizers(),
                use_flash_attn=config.use_flash_attn,
            )
            self.embed_dim=960
        else:
            raise ValueError(f"unknown ESM-C model size: {config.model_size}")

        state_dict = torch.load(
            os.path.join(
                config.weights_path,
                "data",
                "weights",
                f"esmc_{config.model_size}_2024_12_v0.pth",
            ),
        )
        self.model.load_state_dict(state_dict)

        if frozen:
            self.model = self.model.eval()
            self._freeze()
        # Freeze everything after the layer we're
        # using for extracting representations to
        # avoid DDP errors.
        else:
            self._unfreeze(unfreeze_all=False)

        # For masking when calculating original MLM loss
        self.rng = torch.Generator().manual_seed(42)
        self.mask_prob = config.mask_prob

    def _freeze(self):
        for params in self.model.parameters():
            params.requires_grad = False

    def _unfreeze(
        self,
        unfreeze_all: bool = False,
    ):
        for param in self.model.parameters():
            param.requires_grad = True
        if not unfreeze_all:
            # Freeze layers that do not contribute to the embeddings
            # that we're extracting to prevent DDP exceptions.
            num_blocks = len(self.model.transformer.blocks)
            if self.rep_layer != num_blocks-1:
                for block in self.model.transformer.blocks[self.rep_layer+1:]:
                    for param in block.parameters():
                        param.requires_grad = False
            for param in self.model.sequence_head.parameters():
                param.requires_grad = False


    def _get_embedding(
        self,
        protein_tensor: torch.Tensor,
    ) -> torch.Tensor:
        logits_out = self.model(protein_tensor)

        return self.model.transformer.norm(logits_out.hidden_states[self.rep_layer])

    def embed_batch(
        self,
        batch: ESMCBatch,
        protein_level: bool = False,
        zero_non_residue_embeds: bool = False,
    ) -> torch.Tensor:
        """Embed a batch of pre-tokenized protein sequences"""
        residue_embeddings = self._get_embedding(batch.tokenized_seq)

        if protein_level:
            # Return CLS token for protein embedding to be consistent with how ESM-2 protein level
            # embeddings are made in `transformers` implementation of `EsmForSequenceClassification`
            #  https://github.com/huggingface/transformers/blob/main/src/transformers/models/esm/modeling_esm.py#L1028
            return residue_embeddings[:, 0, :]
        else:
            if zero_non_residue_embeds:
                # Make mask that's 1 at every position that corresponds to an actual
                # residue position, 0 otherwise.
                residue_mask = torch.ones_like(batch.tokenized_seq)
                mask = (batch.tokenized_seq == self.model.tokenizer.pad_token_id) \
                       | (batch.tokenized_seq == self.model.tokenizer.eos_token_id) \
                       | (batch.tokenized_seq == self.model.tokenizer.cls_token_id)
                residue_mask.masked_fill_(
                    mask=mask,
                    value=0,
                )
                non_residue_mask = ~(residue_mask.unsqueeze(-1).bool())
                residue_embeddings.masked_fill_(non_residue_mask, 0)
            # Remove CLS token so substructure indices line up
            return residue_embeddings[:, 1:, :]

    # the following two functions are deprecated for the current data module setup
    @torch.inference_mode()
    def embed_single_protein(self, seq: str) -> torch.Tensor:
        """Process a single protein sequence through ESM"""
        protein = ESMProtein(sequence=seq)
        protein_tensor = self.model.encode(protein).sequence.unsqueeze(0)

        seq_len = len(seq) + 2  # +2 for EOS and BOS
        if seq_len <= self.max_len:
            ret = self._get_embedding(protein_tensor)
        else:
            # Get chunk indices for long sequences
            idxs = get_chunk_idxs(seq_len, max_len=self.config.max_seq_length)
            outputs = []
            for start, end in idxs:
                sub_tensor = protein_tensor[:, start:end]
                outputs.append(self._get_embedding(sub_tensor))
            ret = torch.cat(outputs, dim=1)

        return ret.squeeze()[1 : len(seq) + 1]

    @torch.inference_mode()
    def embed_sequences(self, sequences: list[str]) -> list[torch.Tensor]:
        """Embed multiple protein sequences"""
        all_embeddings = []

        for seq in tqdm(sequences, desc="Processing sequences"):
            try:
                embedding = self.embed_single_protein(seq)
                all_embeddings.append(embedding)
            except Exception as e:
                raise e

        return all_embeddings

    def calc_original_loss(
        self,
        batch: ESMCBatch,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Calculate original MLM loss for a batch of tokenized sequences."""
        seqs = batch.tokenized_seq

        ignore_tokens = torch.tensor([
            self.model.tokenizer.pad_token_id,
            self.model.tokenizer.bos_token_id,
            self.model.tokenizer.cls_token_id,
            self.model.tokenizer.eos_token_id,
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

        seqs = seqs.masked_fill(mask, self.model.tokenizer.mask_token_id)

        logits = self.model.forward(seqs).sequence_logits

        # Flatten logits along batch dim and get logits for just masked positions
        masked_pos_logits = logits.reshape(-1, logits.shape[-1])[masked_idxs]
        return CrossEntropyLoss(reduction=reduction)(masked_pos_logits, orig_values_flat)

    def get_embed_dim(self):
        return self.embed_dim

    def model_name(self) -> str:
        return f"ESM-C_{self.model_size}"

    @classmethod
    def get_required_input_type(cls) -> Set[DataType]:
        return {DataType.SEQ}
