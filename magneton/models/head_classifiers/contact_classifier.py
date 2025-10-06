import torch
import torch.nn as nn

from magneton.data.core import Batch
from .interface import HeadModule

class ContactPredictionHead(nn.Module, HeadModule):
    """Head module for contact prediction tasks.
    Args:
        input_dim: The dimension of the inputs, e.g. could be
            embedding dim or stacked attention weights from base model.
        hidden_dims: The dimensions of the hidden layers.
        ignore_index: The index to ignore in the labels.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        ignore_index: int = -1,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.Tanh(),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.contact_head = nn.Sequential(*layers)
        self.ignore_index = ignore_index

    def forward(self, batch: Batch, embedder: nn.Module) -> torch.Tensor:
        with torch.no_grad():
            contact_inputs = embedder.forward_for_contact(batch)

        # These come out of the contact head as (batch, seq_len, seq_len, 1),
        # so squeeze off the last dim.
        return self.contact_head(contact_inputs).squeeze(-1)

    def process_labels(self, batch: Batch, logits: torch.Tensor) -> torch.Tensor:
        # Contact prediction has special label processing
        labels_flat = batch.labels.view(-1)
        keep_idxs = labels_flat != self.ignore_index
        return labels_flat[keep_idxs]

    def get_embed_dim(self, embedder: nn.Module) -> int:
        return embedder.get_attention_dim()
