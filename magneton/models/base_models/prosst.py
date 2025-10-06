from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForMaskedLM, AutoTokenizer

from magneton.data.model_specific.prosst import ProSSTBatch
from magneton.types import DataType

from .interface import BaseConfig, BaseModel
from .utils import get_seq_mask, pool_residue_embeddings


@dataclass(kw_only=True)
class ProSSTConfig(BaseConfig):
    """Configuration options for ProSST models"""

    weights_path: str
    structure_vocab_size: int = 2048
    # Default to final layer hidden states
    rep_layer: int = 12
    mask_prob: float = 0.15


class ProSSTEmbedder(BaseModel):
    "ProSST sequence+structure models."

    def __init__(
        self,
        config: ProSSTConfig,
        frozen: bool = True,
    ):
        super().__init__(config)

        self.vocab_size = config.structure_vocab_size
        self.rep_layer = config.rep_layer
        self.mask_prob = config.mask_prob

        self.model = AutoModelForMaskedLM.from_pretrained(
            config.weights_path, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.weights_path, trust_remote_code=True
        )
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
            if self.rep_layer != num_blocks - 1:
                for block in self.model.prosst.encoder.layer[self.rep_layer + 1 :]:
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

    def calc_original_loss(
        self,
        batch: ProSSTBatch,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Calculate ProSST's original MLM training loss.

        ProSST's MLM loss is a structure-conditioned sequence MLM objective, i.e. we
        mask amino acids but leave the structure tokens unnoised.
        """
        seqs = batch.tokenized_seq

        ignore_tokens = torch.tensor(
            [
                self.tokenizer.pad_token_id,
                self.tokenizer.cls_token_id,
                self.tokenizer.eos_token_id,
            ],
            device=seqs.device,
        )

        mask = get_seq_mask(
            seqs,
            ignore_tokens=ignore_tokens,
            rng=self.rng,
            mask_prob=self.mask_prob,
        )
        masked_idxs = torch.where(mask.reshape(-1))[0]

        # Store original mask values for loss calculation
        orig_values_flat = seqs.reshape(-1)[masked_idxs]

        masked_seqs = seqs.masked_fill(mask, self.tokenizer.mask_token_id)

        attention_mask = torch.ones_like(batch.tokenized_seq)
        attention_mask[batch.tokenized_seq == self.tokenizer.pad_token_id] = 0

        # Not fully sure why, but the structure tokens come through as inference mode
        # tensors. Maybe lightning is working some magic on the train side to make
        # sure this doesn't happen normally
        struct_toks = batch.tokenized_struct.clone().detach()

        logits = self.model(
            input_ids=masked_seqs,
            ss_input_ids=struct_toks,
            attention_mask=attention_mask,
        ).logits

        # Flatten logits along batch dim and get logits for just masked positions
        masked_pos_logits = logits.reshape(-1, logits.shape[-1])[masked_idxs]
        return CrossEntropyLoss(reduction=reduction)(
            masked_pos_logits, orig_values_flat
        )

    def get_embed_dim(self):
        return self.embed_dim

    def model_name(self) -> str:
        return f"ProSST-{self.vocab_size}"

    @classmethod
    def get_required_input_type(cls) -> set[DataType]:
        return {DataType.SEQ, DataType.STRUCT}
