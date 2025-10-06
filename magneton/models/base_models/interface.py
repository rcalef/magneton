from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn

from magneton.data.core import Batch
from magneton.types import DataType


@dataclass(kw_only=True)
class BaseConfig:
    """Config containing parameters shared by all base models."""
    for_contact_prediction: bool = False


class BaseModel(nn.Module, ABC):
    """Interface for base models."""

    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def _freeze(self):
        """Freeze all parameters in the underlying model."""
        pass

    @abstractmethod
    def _unfreeze(self, unfreeze_all: bool = False):
        """Unfreeze parameters in the underlying model.

        When `unfreeze_all` is False, this should only unfreeze
        parameters that are used to calculate the embeddings returned
        by the `embed_batch` method.

        Args:
            - unfreeze_all (bool): Whether to unfreeze all parameters,
                or only those used for calculating returned embeddings.
        """
        pass

    @abstractmethod
    def embed_batch(
        self,
        batch: Batch,
        protein_level: bool,
        zero_non_residue_embeds: bool = False,
    ) -> torch.Tensor:
        """Embed a batch of proteins.

        Args:
            - batch (Batch): the batch of proteins to embed
            - protein_level (bool): Whether to return embeddings
                per protein, or per residue.
            - zero_non_residue_embeds (bool): If returning per-residue
                embeddings, whether or not to zero out embeddings for
                non-residue tokens (e.g. EOS).
        Returns:
            - torch.Tensor of per-protein or per-residue embeddings
        """
        pass

    @abstractmethod
    def forward_for_contact(
        self,
        batch: Batch,
    ) -> torch.Tensor:
        """Return inputs suitable for contact prediction.

        The authors of different models have used different inputs for
        contact prediction (e.g. attention weights vs per-residue embeddings).
        This method allows the base model to return the most appropriate
        input for contact prediction, in case that's distinct from the
        embeddings returned by the `embed_batch` method.

        Args:
            - batch (Batch): the batch of proteins.
        Returns:
            - torch.Tensor of contact prediction inputs
        """
        pass

    @abstractmethod
    def calc_original_loss(
        self,
        batch: Batch,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Calculate original loss used to train base model.

        This function is used when fine-tuning using EWC. This
        should implement the objective originally used for the
        base model (e.g. MLM for ESM-based models). This isn't
        strictly necessary, as it's only used when training with
        EWC.

        Args:
            - batch (Batch): the batch of proteins.
            - reduction (str): Reduction applied to final loss.
        Return:
            - torch.Tensor scalar giving the loss
        """
        pass

    @abstractmethod
    def get_embed_dim(cls) -> int:
        pass

    @classmethod
    @abstractmethod
    def get_required_input_type(cls) -> set[DataType]:
        """Return the required data types for this model"""
        pass

    @classmethod
    @abstractmethod
    def model_name(cls) -> str:
        """Return human-readable name for model"""
        pass
