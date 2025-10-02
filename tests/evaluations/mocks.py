import torch
import torch.nn as nn

from magneton.data.core import Batch
from magneton.data.evals.task_types import TASK_GRANULARITY


class MockEmbedder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(
        self,
        batch: Batch,
        protein_level: bool = False,
        zero_non_residue_embeds: bool = False,
    ) -> torch.Tensor:

        batch_size = len(batch.protein_ids)
        max_len = max(batch.lengths)
        # # Not ideal but easiset for now
        # if hasattr(batch, "tokenized_seq"):
        #     device = batch.tokenized_seq.device
        # elif hasattr(batch, "tokenized_sa_seq"):
        #     device = batch.tokenized_sa_seq.device
        # else:
        #     raise ValueError("this likely needs to be updated for a new model type")

        embeds = torch.randn(
            (batch_size, max_len, self.embed_dim),
            #device=device,
        )
        if protein_level:
            return embeds.mean(dim=1)
        else:
            if zero_non_residue_embeds:
                for i, L in enumerate(batch.lengths):
                    if L < max_len:
                        embeds[i, L:] = 0
            return embeds

    def embed_batch(
        self,
        batch: Batch,
        protein_level: bool = False,
        zero_non_residue_embeds: bool = False,
    ) -> torch.Tensor:
        return self(
            batch=batch,
            protein_level=protein_level,
            zero_non_residue_embeds=zero_non_residue_embeds,
        )

    def forward_for_contact(self, batch: Batch) -> torch.Tensor:
        max_len = max(batch.lengths)
        input_dim = self.get_attention_dim()
        return torch.randn(len(batch.protein_ids), max_len, max_len, input_dim)

    def get_embed_dim(self):
        return self.embed_dim

    def get_attention_dim(self):
        return self.embed_dim

    def _freeze(self):   # match the real interface
        pass

    def _unfreeze(self, unfreeze_all=False):
        pass


class MockEmbeddingMLP:
    """Object mimicking the checkpointed model."""
    def __init__(self):
        self.embedder = MockEmbedder()

    # this is what MultiLabelMLP calls
    @classmethod
    def load_from_checkpoint(cls, *args, **kwargs):
        return cls()

class MockLoader:
    def __init__(self, batches):
        self._batches = list(batches)

    def __iter__(self):
        return iter(self._batches)


class MockDataModule:
    def __init__(self, task_granularity: TASK_GRANULARITY, num_classes: int, batches: list[Batch]):
        self.task_granularity = task_granularity
        self._num_classes = num_classes
        self._val = MockLoader(batches)
        self._test = MockLoader(batches)
        self.distributed = False

    def num_classes(self) -> int:
        return self._num_classes

    def val_dataloader(self):
        return self._val

    def test_dataloader(self):
        return self._test


class MockTrainer:
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, model, datamodule):
        # no-op
        return None

    def save_checkpoint(self, path):
        # no-op
        return None

    def predict(self, model, dataloaders, return_predictions):
        preds = []
        for batch in dataloaders:
            logits = model(batch)
            labels = model.head.process_labels(batch, logits)
            preds.append((logits.detach(), labels.detach()))
        return preds