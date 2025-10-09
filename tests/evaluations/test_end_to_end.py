from pathlib import Path
from unittest.mock import patch

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

from magneton.evaluations.supervised_classification import run_supervised_classification
from ..mocks import MockSubstructureClassifier


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
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="config",
            overrides=[
                f"output_dir={str(tmp_path)}",
                "+evaluate=deepfri",
                f"evaluate.tasks=['{task}']",
                "+evaluate.model_checkpoint='NA'",
                "training.accelerator='cpu'",
                "training.dev_run=true",
                "data.batch_size=2",
            ],
        )
        config = instantiate(cfg)

    with patch(
        "magneton.models.evaluation_classifier.SubstructureClassifier",
        MockSubstructureClassifier,
    ):
        run_supervised_classification(
            config=config, task=task, output_dir=tmp_path, run_id=f"test_{task},"
        )
