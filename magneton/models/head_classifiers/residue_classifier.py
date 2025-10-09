import torch
import torch.nn as nn

from magneton.data.core import Batch
from .interface import HeadModule


class ResidueClassificationHead(nn.Module, HeadModule):
    """Head module for residue-level classification tasks."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dims: list[int],
        num_classes: int,
        dropout_rate: float,
    ):
        super().__init__()
        # Build CNN layers
        layers = []
        prev_dim = embed_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Dropout(dropout_rate),
                    nn.Conv1d(
                        in_channels=prev_dim,
                        out_channels=hidden_dim,
                        kernel_size=5,
                        padding="same",
                    ),
                    nn.Tanh(),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Dropout(dropout_rate))
        self.cnn = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(self, batch: Batch, embedder: nn.Module) -> torch.Tensor:
        # batch_size (num_proteins) X max_len X embed_dim
        residue_embeds = embedder.embed_batch(
            batch,
            protein_level=False,
            zero_non_residue_embeds=True,
        )

        # batch_size X max_len X num_classes
        raw_logits = self._forward_cnn(residue_embeds)

        # Pull out the actual residue logits and flatten
        residue_logits = []
        for idx, length in enumerate(batch.lengths):
            residue_logits.append(raw_logits[idx, :length])

        # total_len X num_classes
        flat_logits = torch.cat(residue_logits)
        return flat_logits

    def _forward_cnn(self, residue_embeds: torch.Tensor) -> torch.Tensor:
        # residue_embeds has dims (batch, max_length, embeds)
        # transpose to (batch, embeds, max_length) for CNN
        x = residue_embeds.transpose(1, 2)
        x = self.cnn(x)
        # transpose back to (batch, max_length, embeds) for final classifier
        x = x.transpose(1, 2)
        return self.classifier(x)

    def get_embed_dim(self, embedder: nn.Module) -> int:
        return embedder.get_embed_dim()
