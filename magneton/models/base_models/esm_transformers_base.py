from dataclasses import dataclass
from pathlib import Path
from importlib.metadata import version
from typing import Literal, Set

import torch
import torch.nn.functional as F
from packaging.version import parse
from transformers import EsmForMaskedLM, EsmTokenizer
from transformers.models.esm.modeling_esm import average_product_correct, symmetrize

from magneton.core_types import DataType
from magneton.data.core import Batch
from magneton.utils import get_model_dir

from .interface import BaseConfig, BaseModel
from .utils import pool_residue_embeddings


def map_esms_keys(k: str) -> str:
    if "layers" in k or "emb_layer_norm_after" in k:
        k = k.replace("q_proj", "query")
        k = k.replace("k_proj", "key")
        k = k.replace("v_proj", "value")
        k = k.replace("self_attn.out_proj", "attention.output.dense")

        k = k.replace("rot_emb", "rotary_embeddings")
        k = k.replace("self_attn_layer_norm", "attention.LayerNorm")
        k = k.replace("self_attn", "attention.self")

        k = k.replace("fc1", "intermediate.dense")
        k = k.replace("fc2", "output.dense")
        k = k.replace("final_layer_norm", "LayerNorm")
        k = k.replace("model", "esm.encoder")
        k = k.replace("layers", "layer")
    if "contact_head" in k:
        k = k.replace("model", "esm")
    if "lm_head" in k:
        k = k.replace("model.", "")
        k = k.replace("lm_head.weight", "lm_head.decoder.weight")

    if k == "model.embed_tokens.weight":
        k = "esm.embeddings.word_embeddings.weight"
    return k

esm_s_weight_map = {
    "esm_150m_s.pth": "esm2_t30_150M_UR50D",
    "esm_650m_s.pth": "esm2_t33_650M_UR50D",
}

@dataclass(kw_only=True)
class TransformersESMBaseConfig(BaseConfig):
    """Config for models using transformers ESM model implementation.

    Args:
        - weights_path (str): Path to directory containing weights from
            HuggingFace.
        - use_flash_attn (bool): Whether or not to use Flash Attention.
        - rep_layer (int): The layer of the model to extract hidden
            representations from as embeddings.
        - mask_prob (float): Mask probability to use when calculating
            original MLM loss.
        - unk_amino_acid_char (str): The character used to represent
            unknown amino acids.
    """

    weights_path: str
    use_flash_attn: bool = False
    rep_layer: int = 12
    mask_prob: float = 0.15
    unk_amino_acid_char: str = "X"


class TransformersESMBaseModel(BaseModel):
    """Base class containing shared logic for models based on transformers ESM implementation."""

    def __init__(
        self,
        config: TransformersESMBaseConfig,
        frozen: bool = True,
    ):
        super().__init__(config)

        self.rep_layer = config.rep_layer
        self.model_size = config.model_size

        weights_path = Path(config.weights_path)
        weights_base = weights_path.name
        # Handle ESM-S weights
        if weights_base in esm_s_weight_map:
            esm_s_state_dict = torch.load(config.weights_path)
            mapped_state_dict = {}
            for k, v in esm_s_state_dict.items():
                mapped_state_dict[map_esms_keys(k)] = v

            esm2_weight_path = get_model_dir() / esm_s_weight_map[weights_base]
            self.model = EsmForMaskedLM.from_pretrained(esm2_weight_path)
            self.model.load_state_dict(mapped_state_dict, strict=False)
            self.tokenizer = EsmTokenizer.from_pretrained(esm2_weight_path)
        # Otherwise just load normally
        else:
            self.model = EsmForMaskedLM.from_pretrained(config.weights_path)
            self.tokenizer = EsmTokenizer.from_pretrained(config.weights_path)
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
        if config.for_contact_prediction:
            self.setup_for_contacts()

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

    def setup_for_contacts(self):
        self.model.config._attn_implementation = "eager"

    def _freeze(self):
        for params in self.model.parameters():
            params.requires_grad = False

    def _unfreeze(
        self,
        unfreeze_all: bool = False,
    ):
        for name, param in self.model.named_parameters():
            # At the risk of causing unintended behavior, we don't want to unfreeze
            # the contact head, since we're not actually tuning this in the general
            # case.
            if "contact_head" in name:
                continue
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

    @torch.inference_mode()
    def forward_for_contact(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        attention_mask = torch.ones_like(tokens)
        attention_mask[tokens == self.tokenizer.pad_token_id] = 0
        out = self.model(
            input_ids=tokens,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        attentions = torch.stack(out.attentions, dim=1)

        # Copied from transformers modeling_esm.py,
        # modified to remove sigmoid so we can use
        # BCEWithLogitsLoss, which is safe with autocast.
        eos_mask = tokens.ne(self.tokenizer.eos_token_id).to(attentions)
        eos_mask = eos_mask.unsqueeze(1) * eos_mask.unsqueeze(2)
        attentions = attentions * eos_mask[:, None, None, :, :]
        attentions = attentions[..., :-1, :-1]
        # remove cls token attentions
        attentions = attentions[..., 1:, 1:]
        batch_size, layers, heads, seqlen, _ = attentions.size()
        attentions = attentions.view(batch_size, layers * heads, seqlen, seqlen)

        # features: batch x channels x tokens x tokens (symmetric)
        attentions = average_product_correct(symmetrize(attentions))
        attentions = attentions.permute(0, 2, 3, 1)

        return attentions

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
        pooling_method: Literal["mean", "cls"] = "mean",
        zero_non_residue_embeds: bool = False,
    ) -> torch.Tensor:
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
            if pooling_method == "mean":
                return pool_residue_embeddings(
                    residue_embeddings, residue_mask=residue_mask
                )
            else:
                return residue_embeddings[:, 0, :]
        else:
            if zero_non_residue_embeds:
                # Make mask that's 1 at every position that corresponds to an actual
                # residue position, 0 otherwise.
                non_residue_mask = ~(residue_mask.unsqueeze(-1).bool())
                residue_embeddings.masked_fill_(non_residue_mask, 0)

            # Remove CLS dimension so substructure indices match
            return residue_embeddings[:, 1:, :]

    def calc_original_loss(
        self,
        batch: Batch,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """NOTE: this modifies the original tokenized seq tensor, call last."""
        raise NotImplementedError("EsmBaseEmbedder is expected to be subclassed")

    def get_embed_dim(self):
        return self.model.config.hidden_size

    def get_attention_dim(self):
        """Get expected dim of stacked attention, used for contact prediction"""
        return (
            self.model.config.num_hidden_layers * self.model.config.num_attention_heads
        )

    def model_name(self) -> str:
        raise NotImplementedError("EsmBaseEmbedder is expected to be subclassed")

    @classmethod
    def get_required_input_type(cls) -> Set[DataType]:
        raise NotImplementedError("EsmBaseEmbedder is expected to be subclassed")
