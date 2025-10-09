from types import SimpleNamespace

import torch

from magneton.config import (
    BaseModelConfig,
    DataConfig,
    ModelConfig,
    PipelineConfig,
    TrainingConfig,
)
from magneton.models import substructure_classifier

from ..mocks import (
    MockBaseModel,
    MockBatch,
    MockSubstructure,
)


def make_simple_config(hidden_dims=None, loss_strategy="none", ewc_weight=1.0):
    """Return a minimal config"""
    if hidden_dims is None:
        hidden_dims = []
    return PipelineConfig(
        model=ModelConfig(
            frozen_base_model=False,
            model_params={"hidden_dims": hidden_dims, "dropout_rate": 0.0},
        ),
        training=TrainingConfig(
            loss_strategy=loss_strategy,
            ewc_weight=ewc_weight,
            reuse_ewc_weights=None,
            learning_rate=1e-3,
            weight_decay=0.0,
            embedding_learning_rate=1e-4,
            embedding_weight_decay=0.0,
        ),
        base_model=BaseModelConfig(model_params={}),
        data=DataConfig(substruct_types=[]),
    )


# ---------------------------
# Tests
# ---------------------------
def test_single_head_forward_and_labels(monkeypatch):
    """
    Single-head (int) case: ensure forward returns logits for all substructures
    and _gather_labels returns labels in the same order.
    """
    embed_dim = 8
    # patch the BaseModelFactory used by the module to return our mock
    monkeypatch.setattr(
        substructure_classifier,
        "BaseModelFactory",
        SimpleNamespace(
            create_base_model=lambda cfg, frozen=False: MockBaseModel(embed_dim)
        ),
    )

    config = make_simple_config(hidden_dims=[16], loss_strategy="none")
    num_classes = 4  # single head -> normalized to {"default": 4}
    model = substructure_classifier.SubstructureClassifier(
        config=config, num_classes=num_classes, load_pretrained_fisher=False
    )

    # build a small batch:
    # protein 0: two substructures (0:2) and (2:4) ; protein 1: one substructure (0:3)
    protein_ids = ["P0", "P1"]
    lengths = [5, 4]
    substructs = [
        [
            MockSubstructure([(0, 2)], "default", 1),
            MockSubstructure([(2, 4)], "default", 2),
        ],
        [MockSubstructure([(0, 3)], "default", 0)],
    ]
    batch = MockBatch(
        protein_ids=protein_ids, lengths=lengths, substructures=substructs
    )

    logits = model.forward(batch)
    assert isinstance(logits, dict)
    assert "default" in logits
    # three substructures total -> logits shape (3, num_classes)
    assert logits["default"].shape == (3, num_classes)

    labels = model._gather_labels(batch)
    assert "default" in labels
    assert labels["default"].shape[0] == 3
    assert labels["default"].dtype == torch.long


def test_multitask_training_step_returns_scalar_loss(monkeypatch):
    """
    Multitask case: verify training_step runs, returns a scalar loss tensor.
    """
    embed_dim = 6
    monkeypatch.setattr(
        substructure_classifier,
        "BaseModelFactory",
        SimpleNamespace(
            create_base_model=lambda cfg, frozen=False: MockBaseModel(embed_dim)
        ),
    )

    config = make_simple_config(hidden_dims=[embed_dim], loss_strategy="none")
    num_classes = {"helix": 3, "sheet": 2}
    model = substructure_classifier.SubstructureClassifier(
        config=config, num_classes=num_classes, load_pretrained_fisher=False
    )

    # build batch with 4 substructures: helix, sheet, helix, sheet
    protein_ids = ["P0", "P1"]
    lengths = [5, 4]
    substructs = [
        [
            MockSubstructure([(0, 2)], "helix", 1),
            MockSubstructure([(2, 4)], "sheet", 0),
        ],
        [
            MockSubstructure([(0, 3)], "helix", 2),
            MockSubstructure([(0, 1)], "sheet", 1),
        ],
    ]
    batch = MockBatch(
        protein_ids=protein_ids, lengths=lengths, substructures=substructs
    )

    loss = model.training_step(batch, batch_idx=0)
    assert torch.is_tensor(loss), "training_step must return a tensor loss"
    # scalar (zero-dimensional tensor)
    assert loss.dim() == 0


def test_ewc_penalty_is_added_correctly(monkeypatch):
    """
    Configure model with EWC and a synthetic fisher_info buffer; verify that
    the computed training_step loss equals base_loss + ewc_weight * ewc_term.
    """
    embed_dim = 5
    vocab_size = 5
    monkeypatch.setattr(
        substructure_classifier,
        "BaseModelFactory",
        SimpleNamespace(
            create_base_model=lambda cfg, frozen=False: MockBaseModel(
                embed_dim=embed_dim,
                vocab_size=5,
            )
        ),
    )

    ewc_weight = 0.1
    config = make_simple_config(
        hidden_dims=[embed_dim],
        loss_strategy="ewc",
        ewc_weight=ewc_weight,
    )
    num_classes = {"default": 3}
    model = substructure_classifier.SubstructureClassifier(
        config=config,
        num_classes=num_classes,
        load_pretrained_fisher=True,
    )

    # Make a tiny batch with 2 substructures
    protein_ids = ["P0"]
    lengths = [4]
    substructs = [
        [
            MockSubstructure([(0, 2)], "default", 1),
            MockSubstructure([(2, 4)], "default", 2),
        ]
    ]
    tokens = torch.randint(low=0, high=vocab_size, size=(2, 4))
    batch = MockBatch(
        protein_ids=protein_ids,
        lengths=lengths,
        substructures=substructs,
        tokens=tokens,
    )

    # Ensure an original_params vector (typically set in on_train_start)
    curr_params_vec = torch.nn.utils.parameters_to_vector(model.base_model.parameters())
    # set original params to zeros to make the difference equal curr_params_vec
    model.original_params = torch.zeros_like(curr_params_vec)

    # populate fisher_info buffer with ones (same shape as curr_params_vec)
    model.fisher_info.fill_(1.0)

    # compute expected base loss manually
    logits = model.forward(batch)
    base_losses = []
    labels_per = model._gather_labels(batch)
    for h, head_logits in logits.items():
        base_losses.append(model.loss(head_logits, labels_per[h]))
    print(f"got: {base_losses}")
    base_total = torch.stack(base_losses).sum()

    # calculate expected ewc term
    curr_params_vec = torch.nn.utils.parameters_to_vector(model.base_model.parameters())
    ewc_loss = (
        (curr_params_vec - model.original_params).pow(2) * model.fisher_info
    ).sum()
    expected_total = base_total + ewc_weight * ewc_loss

    # run training_step and compare
    returned = model.training_step(batch, batch_idx=0)
    assert torch.is_tensor(returned)
    # numeric closeness
    assert torch.allclose(
        returned.detach(), expected_total.detach(), rtol=1e-4, atol=1e-6
    ), f"expected {expected_total.item():.6f}, got {returned.item():.6f}"
