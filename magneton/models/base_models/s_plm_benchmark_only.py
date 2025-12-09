import sys
import yaml

from dataclasses import dataclass
from pathlib import Path
from typing import Set

import torch
import torch.nn.functional as F

from magneton.core_types import DataType
from magneton.data.core import Batch
from magneton.data.model_specific.esm2 import ESM2Batch

from .interface import BaseConfig, BaseModel


# NOTE: To use S-PLM within Magneton, you will have to apply
# the git patch file contained within this directory to the S-PLM
# submodule:
#   cd /path/to/magneton/magneton/external/S-PLM
#   git apply ../../models/base_models/s_plm_compatibility.patch
#
S_PLM_PATH = Path(__file__).parent.parent.parent / "external" / "S-PLM"
sys.path.append(str(S_PLM_PATH))
from splm_utils import load_configs, load_checkpoints_only
from model import SequenceRepresentation

@dataclass(kw_only=True)
class SPLMConfig(BaseConfig):
    """Config for models using S-PLM.

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


class SPLMBaseModel(BaseModel):
    """Base class containing shared logic for models based on transformers ESM implementation."""

    def __init__(
        self,
        config: SPLMConfig,
        frozen: bool = True,
    ):
        super().__init__(config)
        # Load the configuration file
        config_path = S_PLM_PATH / "configs" / "representation_config.yaml"
        with open(config_path) as f:
            dict_config = yaml.full_load(f)
        configs = load_configs(dict_config)

        # Create the model using the configuration file
        model = SequenceRepresentation(logging=None, configs=configs)
        load_checkpoints_only(config.weights_path, model)

        self.model = model
        self._freeze()
        self.model.eval()

    def setup_for_contacts(self):
        raise NotImplementedError("not implemented")

    def _freeze(self):
        for params in self.model.parameters():
            params.requires_grad = False

    def _unfreeze(
        self,
        unfreeze_all: bool = False,
    ):
        raise NotImplementedError("not implemented")

    @torch.inference_mode()
    def forward_for_contact(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError("not implemented")

    def _get_embedding(
        self,
        protein_tensor: torch.Tensor,
    ) -> torch.Tensor:
        out = self.model(
            protein_tensor,
            output_hidden_states=True,
        )

        return F.normalize(out.hidden_states[self.rep_layer], dim=-1)

    def embed_batch(
        self,
        batch: ESM2Batch,
        protein_level: bool = False,
        zero_non_residue_embeds: bool = False,
    ) -> torch.Tensor:
        """Embed a batch of pre-tokenized protein sequences"""
        tokens = batch.tokenized_seq
        protein_representation, residue_representation, mask = self.model(tokens)
        if protein_level:
            return protein_representation
        else:
            if zero_non_residue_embeds:
                # Make mask that's 1 at every position that corresponds to an actual
                # residue position, 0 otherwise.
                non_residue_mask = ~(mask.unsqueeze(-1).bool())
                residue_representation.masked_fill_(non_residue_mask, 0)

            return residue_representation

    def calc_original_loss(
        self,
        batch: Batch,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """NOTE: this modifies the original tokenized seq tensor, call last."""
        return 0

    def get_embed_dim(self):
        return self.model.esm2.embed_dim

    def get_attention_dim(self):
        """Get expected dim of stacked attention, used for contact prediction"""
        raise NotImplementedError("not implemented")
        return (
            self.model.config.num_hidden_layers * self.model.config.num_attention_heads
        )

    def model_name(self) -> str:
        return "S-PLM"

    @classmethod
    def get_required_input_type(cls) -> Set[DataType]:
        return {DataType.SEQ}
