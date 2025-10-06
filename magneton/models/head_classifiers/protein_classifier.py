import torch
import torch.nn as nn

from magneton.data.core import Batch
from .interface import HeadModule

class ProteinClassificationHead(nn.Module, HeadModule):
    """Head module for protein-level classification tasks."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dims: list[int],
        num_classes: int,
        dropout_rate: float,
    ):
        super().__init__()
        # Build MLP layers
        layers = []
        prev_dim = embed_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Dropout(dropout_rate),
                    nn.Linear(prev_dim, hidden_dim),
                    nn.Tanh(),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(prev_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, batch: Batch, embedder: nn.Module) -> torch.Tensor:
        # batch_size (num_proteins) X embed_dim
        protein_embeds = embedder.embed_batch(batch, protein_level=True)
        # proteins X num_classes
        return self.mlp(protein_embeds)

    def get_embed_dim(self, embedder: nn.Module) -> int:
        return embedder.get_embed_dim()
