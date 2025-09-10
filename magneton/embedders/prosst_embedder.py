
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from transformers import AutoModelForMaskedLM, AutoTokenizer

from magneton.data.model_specific.prosst import ProSSTBatch
from magneton.types import DataType

from .base_embedder import BaseConfig,  BaseEmbedder
from .utils import pool_residue_embeddings


@dataclass
class ProSSTConfig(BaseConfig):
    weights_path: str = field(kw_only=True)
    structure_vocab_size: int = field(kw_only=True, default=2048)
    # Default to final layer hidden states
    rep_layer: int = field(kw_only=True, default=12)


class ProSSTEmbedder(BaseEmbedder):
    def __init__(
        self,
        config: ProSSTConfig,
        frozen: bool = True,
    ):
        super().__init__(config)

        self.vocab_size = config.structure_vocab_size
        self.rep_layer = config.rep_layer

        self.model = AutoModelForMaskedLM.from_pretrained(config.weights_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(config.weights_path, trust_remote_code=True)
        self.embed_dim = self.model.get_output_embeddings().in_features

        if frozen:
            self.model = self.model.eval()
            self._freeze()
        else:
            self._unfreeze(unfreeze_all=False)

        # For masking when calculating original MLM loss
        self.rng = torch.Generator().manual_seed(42)

    def _freeze(self):
        for params in self.model.parameters():
            params.requires_grad = False

    def _unfreeze(
        self,
        unfreeze_all: bool = False,
    ):
        for name, param in self.model.named_parameters():
            param.requires_grad = True

        if not unfreeze_all:
            # Freeze layers that do not contribute to the embeddings
            # that we're extracting to prevent DDP exceptions.
            num_blocks = len(self.model.prosst.encoder.layer)
            if self.rep_layer != num_blocks-1:
                for block in self.model.prosst.encoder.layer[self.rep_layer+1:]:
                    for param in block.parameters():
                        param.requires_grad = False
            for param in self.model.cls.parameters():
                param.requires_grad = False

        # Not sure if this is a bug in ProSST or not, but these layers never get used.
        # See https://github.com/ai4protein/ProSST/issues/15 for details
        for name, param in self.model.named_parameters():
            if "ss_q_proj" in name:
                param.requires_grad = False

    def embed_batch(
        self,
        batch: ProSSTBatch,
        protein_level: bool = False,
        zero_non_residue_embeds: bool = False,
    ) -> torch.Tensor:
        """Embed a batch of pre-tokenized protein sequences"""
        attention_mask = torch.ones_like(batch.tokenized_seq)
        attention_mask[batch.tokenized_seq == self.tokenizer.pad_token_id] = 0

        out = self.model(
            input_ids=batch.tokenized_seq,
            ss_input_ids=batch.tokenized_struct,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Normalize along the embedding dimension, don't remove BOS token embedding
        residue_embeddings = F.normalize(out.hidden_states[self.rep_layer], dim=-1)
        if protein_level:
            # Mean pool everything except pad tokens, i.e. including BOS and EOS
            return pool_residue_embeddings(residue_embeddings, attention_mask)
        else:
            if zero_non_residue_embeds:
                non_residue_mask = ~(attention_mask.unsqueeze(-1).bool())
                residue_embeddings.masked_fill_(non_residue_mask, 0)
            # Remove CLS token so substructure indices line up
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
        batch: ProSSTBatch,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """NOTE: this modifies the original tokenized seq tensor, call last."""
        return 0

    def get_embed_dim(self):
        return self.embed_dim

    def model_name(self) -> str:
        return f"ProSST-{self.vocab_size}"

    @classmethod
    def get_required_input_type(cls) -> set[DataType]:
        return {DataType.SEQ, DataType.STRUCT}
