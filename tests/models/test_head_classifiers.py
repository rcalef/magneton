import torch

from magneton.data.core.unified_dataset import Batch
from magneton.models.head_classifiers import (
    ProteinClassificationHead,
    ResidueClassificationHead,
    ContactPredictionHead,
    PPIPredictionHead,
)
from ..mocks import MockBaseModel


def make_batch(num_items: int, lengths: list[int], labels: torch.Tensor) -> Batch:
    protein_ids = [f"P{i}" for i in range(num_items)]
    return Batch(
        protein_ids=protein_ids,
        lengths=lengths,
        labels=labels,
        seqs=["A" * L for L in lengths],
    )


def test_protein_head_forward_and_labels():
    embed_dim = 8
    num_classes = 5
    head = ProteinClassificationHead(embed_dim, hidden_dims=[embed_dim], num_classes=num_classes, dropout_rate=0.0)
    embedder = MockBaseModel(embed_dim)

    batch = make_batch(num_items=3, lengths=[10, 12, 8], labels=torch.randint(0, 2, (3, num_classes)).float())
    logits = head.forward(batch, embedder)
    assert logits.shape == (3, num_classes)

    processed = head.process_labels(batch, logits)
    assert processed.shape == (3, num_classes)


def test_residue_head_forward_and_labels():
    embed_dim = 6
    num_classes = 3
    head = ResidueClassificationHead(embed_dim, hidden_dims=[embed_dim], num_classes=num_classes, dropout_rate=0.0)
    embedder = MockBaseModel(embed_dim)

    lengths = [4, 7]
    total_len = sum(lengths)
    labels = torch.randint(0, num_classes, (total_len,))
    batch = make_batch(num_items=2, lengths=lengths, labels=labels)

    logits = head.forward(batch, embedder)
    assert logits.shape == (total_len, num_classes)

    processed = head.process_labels(batch, logits)
    assert processed.shape == (total_len,)


def test_contact_head_forward_and_labels():
    input_dim = 4
    head = ContactPredictionHead(input_dim=input_dim, hidden_dims=[8])
    embedder = MockBaseModel(embed_dim=input_dim)

    lengths = [5]
    L = lengths[0]
    labels = torch.randint(0, 2, (1, L, L)).float()
    # mask out diagonal as -1 to simulate ignored labels
    labels[:, torch.arange(L), torch.arange(L)] = -1
    batch = make_batch(num_items=1, lengths=lengths, labels=labels)

    logits = head.forward(batch, embedder)
    assert logits.shape == (1, L, L)

    processed = head.process_labels(batch, logits)
    # should exclude -1 entries
    assert processed.numel() == (labels != -1).sum().item()


def test_ppi_head_forward_and_labels():
    embed_dim = 5
    head = PPIPredictionHead(embed_dim=embed_dim, hidden_dims=[embed_dim], dropout_rate=0.0)
    embedder = MockBaseModel(embed_dim)

    lengths = [10, 12, 9, 7]
    # labels duplicated per pair: [y0, y0, y1, y1]
    y = torch.tensor([1.0, 0.0])
    labels = torch.stack([y[0], y[0], y[1], y[1]])
    batch = make_batch(num_items=4, lengths=lengths, labels=labels)

    logits = head.forward(batch, embedder)
    assert logits.shape == (2, 1)

    processed = head.process_labels(batch, logits)
    assert processed.shape == (2,)
    # labels should be float and pair-collapsed
    assert torch.allclose(processed, y)


