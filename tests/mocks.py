import torch
import torch.nn as nn

from magneton.core_types import DataType
from magneton.data.core import Batch
from magneton.data.evaluations.task_types import TASK_GRANULARITY
from magneton.models.base_models import BaseConfig, BaseModel


class MockBaseModel(BaseModel):
    def __init__(
        self,
        embed_dim: int = 4,
        vocab_size: int = 4,
    ):
        super().__init__(BaseConfig())
        self.embed_dim = embed_dim

        # Add some actual layers to be used if needed
        self.layers = nn.Sequential(
            nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        batch: Batch,
        protein_level: bool = False,
        zero_non_residue_embeds: bool = False,
    ) -> torch.Tensor:
        batch_size = len(batch.protein_ids)
        max_len = max(batch.lengths)

        if hasattr(batch, "tokens") and batch.tokens is not None:
            embeds = self.layers(batch.tokens)
        else:
            embeds = torch.randn(
                (batch_size, max_len, self.embed_dim),
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

    def _freeze(self):  # match the real interface
        pass

    def _unfreeze(self, unfreeze_all=False):
        pass

    def calc_original_loss(self, batch: Batch, reduction: str = "sum") -> torch.Tensor:
        return torch.Tensor([1])

    @classmethod
    def get_required_input_type(cls) -> set[DataType]:
        return {}

    @classmethod
    def model_name(cls) -> str:
        return "mock"


class MockSubstructureClassifier:
    """Object mimicking the checkpointed model."""

    def __init__(self):
        self.base_model = MockBaseModel()

    # this is what EvaluationClassifier calls
    @classmethod
    def load_from_checkpoint(cls, *args, **kwargs):
        return cls()


class MockSubstructure:
    def __init__(self, ranges, element_type, label):
        self.ranges = ranges  # list of (start,end)
        self.element_type = element_type
        self.label = label


class MockBatch:
    def __init__(
        self,
        protein_ids: list[str],
        lengths: list[int],
        substructures: list[list[tuple[int]]],
        tokens: torch.Tensor | None = None,
    ):
        """
        substructures: list (per-protein) of list(substructure objects)
        lengths: list of ints (same len as protein_ids)
        """
        self.protein_ids = protein_ids
        self.lengths = lengths
        self.substructures = substructures
        self.tokens = tokens


class MockLoader:
    def __init__(self, batches):
        self._batches = list(batches)

    def __iter__(self):
        return iter(self._batches)


class MockDataModule:
    def __init__(
        self, task_granularity: TASK_GRANULARITY, num_classes: int, batches: list[Batch]
    ):
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
