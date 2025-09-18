from dataclasses import dataclass
from importlib.metadata import version
from typing import Set

import torch
import torch.nn.functional as F
from packaging.version import parse
from transformers import EsmForMaskedLM, EsmTokenizer

from magneton.data.model_specific.saprot import SaProtBatch
from magneton.types import DataType

from .base_embedder import BaseConfig, BaseEmbedder
from .utils import pool_residue_embeddings


@dataclass(kw_only=True)
class ESMBaseConfig(BaseConfig):
    weights_path: str
    use_flash_attn: bool = False
    rep_layer: int = 12
    max_seq_length: int = 2048
    mask_prob: float = 0.15


class ESMBaseEmbedder(BaseEmbedder):
    def __init__(
        self,
        config: ESMBaseConfig,
        frozen: bool = True,
    ):
        super().__init__(config)

        self.max_len = config.max_seq_length
        self.rep_layer = config.rep_layer
        self.model_size = config.model_size

        self.tokenizer = EsmTokenizer.from_pretrained(config.weights_path)
        self.model = EsmForMaskedLM.from_pretrained(config.weights_path)
        if config.use_flash_attn:
            self.model.config._attn_implementation = "flash_attention_2"
            # Flash attention support was only added in 4.56.1
            installed_version_str = version("transformers")
            installed_version = parse(installed_version_str)
            required_version = parse("4.56.1")
            if installed_version < required_version:
                raise RuntimeError(
                    f"flash attention with SaProt requires transformers >= 4.56.1, found: {installed_version_str}"
                )

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
            num_blocks = len(self.model.esm.encoder.layer)

            # Returned hidden states are actually one more than the
            # number of blocks since there's a final layer norm after
            # the transformer stack that adds one more set of hidden states.
            if self.rep_layer < num_blocks - 1:
                for block in self.model.esm.encoder.layer[self.rep_layer - 1 :]:
                    for param in block.parameters():
                        param.requires_grad = False
            if self.rep_layer < num_blocks:
                for param in self.model.esm.encoder.emb_layer_norm_after.parameters():
                    param.requires_grad = False
            for param in self.model.esm.contact_head.parameters():
                param.requires_grad = False
            for param in self.model.lm_head.parameters():
                param.requires_grad = False

    def _get_embedding(
        self,
        protein_tensor: torch.Tensor,
    ) -> torch.Tensor:
        out = self.model(
            protein_tensor,
            output_hidden_states=True,
        )

        return F.normalize(out.hidden_states[self.rep_layer], dim=-1)

    def _embed_batch(
        self,
        token_tensor: torch.Tensor,
        protein_level: bool = False,
        zero_non_residue_embeds: bool = False,
    ) -> torch.Tensor:
        """Embed a batch of pre-tokenized protein sequences"""
        """Embed a batch of pre-tokenized protein sequences"""
        attention_mask = torch.ones_like(token_tensor)
        attention_mask[token_tensor == self.tokenizer.pad_token_id] = 0

        out = self.model(
            input_ids=token_tensor,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        residue_embeddings = F.normalize(out.hidden_states[self.rep_layer], dim=-1)

        # Match SaProt code by excluding CLS
        residue_mask = torch.ones_like(token_tensor)
        mask = (
            (token_tensor == self.tokenizer.pad_token_id)
            | (token_tensor == self.tokenizer.eos_token_id)
            | (token_tensor == self.tokenizer.cls_token_id)
        )
        residue_mask.masked_fill_(
            mask=mask,
            value=0,
        )

        if protein_level:
            return pool_residue_embeddings(
                residue_embeddings, residue_mask=residue_mask
            )
        else:
            if zero_non_residue_embeds:
                # Make mask that's 1 at every position that corresponds to an actual
                # residue position, 0 otherwise.
                non_residue_mask = ~(residue_mask.unsqueeze(-1).bool())
                residue_embeddings.masked_fill_(non_residue_mask, 0)

            # Remove CLS dimension so substructure indices match
            return residue_embeddings[:, 1:, :]

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
        """NOTE: this modifies the original tokenized seq tensor, call last."""
        raise NotImplementedError("EsmBaseEmbedder is expected to be subclassed")

    def get_embed_dim(self):
        return self.model.config.hidden_size

    def model_name(self) -> str:
        raise NotImplementedError("EsmBaseEmbedder is expected to be subclassed")

    @classmethod
    def get_required_input_type(cls) -> Set[DataType]:
        raise NotImplementedError("EsmBaseEmbedder is expected to be subclassed")


def get_seq_mask(
    tokenized_seqs: torch.Tensor,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    rng: torch.Generator,
    mask_prob: float = 0.15,
) -> torch.Tensor:
    probs = torch.rand(
        tokenized_seqs.shape,
        generator=rng,
    ).to(tokenized_seqs.device)
    mask = (
        (probs < mask_prob)
        & (tokenized_seqs != pad_token_id)
        & (tokenized_seqs != bos_token_id)
        & (tokenized_seqs != eos_token_id)
    )
    return mask
