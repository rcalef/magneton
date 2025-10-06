from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from magneton.data.core import Batch


class HeadModule(ABC):
    """Abstract base class for task-specific head modules."""

    @abstractmethod
    def forward(self, batch: Batch, embedder: nn.Module) -> torch.Tensor:
        """Forward pass through the head module."""
        pass

    def process_labels(self, batch: Batch, logits: torch.Tensor) -> torch.Tensor:
        """Implement any task-specific label processing.

        Args:
            batch: The batch of data.
            logits: The logits from the head module.

        Returns:
            The processed labels.
        This is used for more bespoke label processing, e.g. deduplicating labels
        for protein pairs in PPI, or flattening the labels for contact prediction.
        Processing for different task types (e.g. binary vs multiclass) is handled
        separately.
        """
        return batch.labels

    @abstractmethod
    def get_embed_dim(self, embedder: nn.Module) -> int:
        """Get the required embedding dimension for this head."""
        pass
