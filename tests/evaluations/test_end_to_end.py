from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

from magneton.evals.supervised_classification import run_supervised_classification
from .mocks import MockEmbeddingMLP


# @pytest.mark.parametrize("task", ["GO:MF", "EC", "fold", "solubility", "fluorescence"])
@pytest.mark.parametrize(
    "task",
    [
        "saprot_binloc",
        "saprot_thermostability",
        "saprot_subloc",
        "human_ppi",
        "FLIP_bind",
    ],
)
def test_multilabelmlp_with_dummy_embedder(tmp_path, task):
    GlobalHydra.instance().clear()
    config_dir = Path(__file__).parent.parent.parent / "magneton" / "configs"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="config",
            overrides=[
                "+evaluate=deepfri",
                f"evaluate.tasks=['{task}']",
                "+evaluate.model_checkpoint='NA'",
                f"training.accelerator='{device}'",
                "training.dev_run=true",
                "data.batch_size=2",
            ],
        )
        config = instantiate(cfg)

    with patch("magneton.evals.downstream_classifiers.EmbeddingMLP", MockEmbeddingMLP):
        run_supervised_classification(
            config=config, task=task, output_dir=tmp_path, run_id=f"test_{task},"
        )
