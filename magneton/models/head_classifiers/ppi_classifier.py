import torch
import torch.nn as nn

from magneton.data.core import Batch
from .interface import HeadModule


class PPIPredictionHead(nn.Module, HeadModule):
    """Head module for protein-protein interaction prediction tasks."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dims: list[int],
        dropout_rate: float,
    ):
        super().__init__()
        # Embed dim is here is 2 x model_embed_dim, since we just concatenate
        # the embeddings of the two proteins for input to the predictor head
        input_dim = embed_dim * 2

        layers = []
        prev_dim = input_dim
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
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, batch: Batch, embedder: nn.Module) -> torch.Tensor:
        assert len(batch.protein_ids) % 2 == 0, f"got: {batch}"
        # batch_size (num_proteins) X embed_dim
        protein_embeds = embedder.embed_batch(batch, protein_level=True)

        # Each pair of adjacent elements is one PPI pair, so want to
        # concatenate those
        num_proteins, embed_dim = protein_embeds.shape
        num_ppis = num_proteins // 2

        # (num_ppis, 2*embed_dim)
        ppi_embeds = protein_embeds.view(num_ppis, 2, embed_dim).flatten(1)

        # (num_ppis, 1)
        return self.mlp(ppi_embeds)

    def process_labels(self, batch: Batch, logits: torch.Tensor) -> torch.Tensor:
        # PPI prediction has special label processing
        assert len(batch.labels) % 2 == 0, f"got: {batch}"
        labels = batch.labels[::2]
        labels_check = batch.labels[1::2]
        assert (labels == labels_check).all(), f"got: {batch}"

        # (num_ppis, )
        return labels

    def get_embed_dim(self, embedder: nn.Module) -> int:
        return embedder.get_embed_dim()
