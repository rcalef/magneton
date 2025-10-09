import json
from types import SimpleNamespace

import pytest
import torch

from magneton.config import (
    BaseModelConfig,
    DataConfig,
    EvalConfig,
    ModelConfig,
    PipelineConfig,
    TrainingConfig,
)
from magneton.data.core.unified_dataset import Batch
from magneton.data.evaluations.task_types import TASK_GRANULARITY
from magneton.evaluations import supervised_classification

from ..mocks import MockBaseModel, MockDataModule, MockTrainer


@pytest.mark.parametrize(
    "task,granularity,num_classes,label_builder",
    [
        (
            "EC",
            TASK_GRANULARITY.PROTEIN_CLASSIFICATION,
            7,
            lambda n, Ls: torch.randint(0, 2, (n, 7)),
        ),
    ],
)
def test_run_supervised_classification_minimal(
    monkeypatch, tmp_path, task, granularity, num_classes, label_builder
):
    # Monkeypatch Trainer
    monkeypatch.setattr(supervised_classification.L, "Trainer", MockTrainer)

    # Monkeypatch EmbeddingMLP.load_from_checkpoint to return stub with embedder
    from magneton.models import substructure_classifier as sc

    def fake_load_from_checkpoint(*args, **kwargs):
        return SimpleNamespace(base_model=MockBaseModel(embed_dim=8))

    monkeypatch.setattr(
        sc.SubstructureClassifier,
        "load_from_checkpoint",
        staticmethod(fake_load_from_checkpoint),
    )

    # Monkeypatch SupervisedDownstreamTaskDataModule to a tiny fake
    lengths = [5, 8, 4]
    labels = label_builder(len(lengths), lengths)
    batches = [
        Batch(
            protein_ids=[f"P{i}" for i in range(len(lengths))],
            lengths=lengths,
            labels=labels,
            seqs=["A" * L for L in lengths],
        )
    ]

    def fake_module(*args, **kwargs):
        return MockDataModule(granularity, num_classes, batches)

    monkeypatch.setattr(
        supervised_classification, "SupervisedDownstreamTaskDataModule", fake_module
    )

    # Build minimal config
    cfg = PipelineConfig(
        output_dir=str(tmp_path),
        run_id="testrun",
        data=DataConfig(data_dir=str(tmp_path), batch_size=2),
        base_model=BaseModelConfig(model="esm2", model_params={}),
        model=ModelConfig(
            model_params={"hidden_dims": ["embed"], "dropout_rate": 0.0},
            frozen_base_model=True,
        ),
        training=TrainingConfig(
            max_epochs=1,
            accelerator="cpu",
            strategy="auto",
            precision="32-true",
            devices=1,
            dev_run=True,
        ),
        evaluate=EvalConfig(
            tasks=[task],
            data_dir=str(tmp_path),
            model_checkpoint=str(tmp_path / "dummy_embed.ckpt"),
            final_prediction_only=False,
            rerun_completed=True,
        ),
    )

    out_dir = tmp_path / task
    out_dir.mkdir(exist_ok=True)

    supervised_classification.run_supervised_classification(
        config=cfg, task=task, output_dir=out_dir, run_id="testrun"
    )

    # Validate that metrics files were produced
    valid_metrics = out_dir / f"validation_{task}_metrics.json"
    test_metrics = out_dir / f"test_{task}_metrics.json"
    assert valid_metrics.exists(), f"Missing {valid_metrics}"
    assert test_metrics.exists(), f"Missing {test_metrics}"

    # Basic JSON sanity
    with open(valid_metrics) as f:
        data = json.load(f)
        assert data["task"] == task
