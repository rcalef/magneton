from itertools import cycle
from typing import Any

import fire
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from tqdm import tqdm

from magneton.config import PipelineConfig
from magneton.data import (
    MagnetonDataModule,
    SupervisedDownstreamTaskDataModule,
)
from magneton.data.core import Batch, get_substructure_parser
from magneton.data.evaluations import (
    EVAL_TASK,
    TASK_TO_TYPE,
)
from magneton.models.evaluation_classifier import (
    _create_head_module,
    _get_labels_for_loss,
)
from magneton.models.substructure_classifier import SubstructureClassifier
from magneton.models.utils import parse_hidden_dims

device = "cuda" if torch.cuda.is_available() else "cpu"


def prep_data_modules(
    config: PipelineConfig,
    eval_task: str,
    num_workers: int = 16,
) -> tuple[torch.utils.data.DataLoader]:
    model_type = config.base_model.model
    magneton_module = MagnetonDataModule(
        data_config=config.data,
        model_type=model_type,
        num_workers=num_workers,
    )

    eval_module = SupervisedDownstreamTaskDataModule(
        data_config=config.data,
        task=eval_task,
        data_dir=config.evaluate.data_dir,
        model_type=model_type,
        unk_amino_acid_char=config.base_model.model_params.get(
            "unk_amino_acid_char", "X"
        ),
        num_workers=num_workers,
    )
    return magneton_module, eval_module


def prep_models(
    config: PipelineConfig,
    eval_task: str,
    eval_module: SupervisedDownstreamTaskDataModule,
) -> dict[Any]:
    substruct_parser = get_substructure_parser(config.data)
    model = SubstructureClassifier(
        config=config,
        num_classes=substruct_parser.num_labels(),
    )
    model.base_model._unfreeze()

    task_type = TASK_TO_TYPE[eval_task]
    task_granularity = eval_module.task_granularity
    embed_dim = model.base_model.get_embed_dim()
    hidden_dims = parse_hidden_dims(
        raw_dims=config.model.model_params["hidden_dims"], embed_dim=embed_dim
    )
    num_classes = eval_module.num_classes()

    eval_head = _create_head_module(
        task_granularity=task_granularity,
        embed_dim=embed_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout_rate=config.model.model_params["dropout_rate"],
    )

    if task_type == EVAL_TASK.MULTILABEL:
        eval_loss = nn.BCEWithLogitsLoss()
    elif task_type == EVAL_TASK.MULTICLASS:
        eval_loss = nn.CrossEntropyLoss()
    elif task_type == EVAL_TASK.BINARY:
        eval_loss = nn.BCEWithLogitsLoss()
    elif task_type == EVAL_TASK.REGRESSION:
        eval_loss = nn.MSELoss()

    return {
        "substruct_model": model,
        "head_model": eval_head,
        "task_type": task_type,
        "eval_loss": eval_loss,
    }


def _extract_base_grads(
    model: SubstructureClassifier,
) -> torch.Tensor:
    all_grads = []
    for params in model.base_model.parameters():
        if params.requires_grad and params.grad is not None:
            all_grads.append(torch.flatten(params.grad.detach()))
    all_grads = torch.cat(all_grads)
    all_grads = all_grads / torch.norm(all_grads)

    model.zero_grad()
    return all_grads


def get_substruct_grads(
    model: SubstructureClassifier,
    batch: Batch,
) -> torch.Tensor:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        substruct_loss = model.training_step(batch, 0)
    substruct_loss.backward()

    return _extract_base_grads(model)


def get_eval_grads(
    model: SubstructureClassifier,
    head_model: nn.Module,
    batch: Batch,
    loss_fn: nn.Module,
    task_type: EVAL_TASK,
) -> torch.Tensor:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = head_model.forward(batch, model.base_model)
        labels = head_model.process_labels(batch, logits)
        labels = _get_labels_for_loss(labels, task_type, logits.dtype)
        loss = loss_fn(logits, labels)
    loss.backward()

    return _extract_base_grads(model)


def run_one_analysis(
    config: PipelineConfig,
    eval_task: str,
    tot_iters: int,
    num_workers: int = 16,
) -> list[float]:
    magneton_module, eval_module = prep_data_modules(
        config=config,
        eval_task=eval_task,
        num_workers=num_workers,
    )

    magneton_loader = magneton_module.train_dataloader()
    eval_loader = eval_module.train_dataloader()

    model_parts = prep_models(
        config,
        eval_task=eval_task,
        eval_module=eval_module,
    )

    substruct_model = model_parts["substruct_model"]
    head_model = model_parts["head_model"]
    eval_task_type = model_parts["task_type"]
    eval_loss_fn = model_parts["eval_loss"]

    substruct_model = substruct_model.to(device)
    head_model = head_model.to(device)

    magneton_iter = cycle(magneton_loader)
    eval_iter = cycle(eval_loader)

    cross_sims = []
    self_substruct_sims = []
    self_eval_sims = []

    prev_substruct_grad = None
    prev_eval_grad = None

    num_iters = 0
    for _ in tqdm(range(tot_iters)):
        substruct_batch = next(magneton_iter)
        eval_batch = next(eval_iter)
        substruct_batch = substruct_batch.to(device)
        eval_batch = eval_batch.to(device)

        substruct_grads = get_substruct_grads(substruct_model, substruct_batch)
        eval_grads = get_eval_grads(
            substruct_model,
            head_model=head_model,
            batch=eval_batch,
            loss_fn=eval_loss_fn,
            task_type=eval_task_type,
        )

        sim = eval_grads @ substruct_grads
        cross_sims.append(sim.item())
        if prev_substruct_grad is not None:
            sim = substruct_grads @ prev_substruct_grad
            self_substruct_sims.append(sim.item())
        if prev_eval_grad is not None:
            sim = eval_grads @ prev_eval_grad
            self_eval_sims.append(sim.item())
        prev_substruct_grad = substruct_grads
        prev_eval_grad = eval_grads

        num_iters += 1

    return cross_sims, self_substruct_sims, self_eval_sims


def run(
    eval_task: str,
    output_path: str,
    model_type: str = "esm2_150m",
    tot_iters: int = 1000,
    num_workers: int = 16,
):
    config_dir = "/home/rcalef/sandbox/repos/magneton/magneton/configs"
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                f"base_model='{model_type}'",
                "+evaluate=deepfri",
                "output_dir='/tmp/'",
                "evaluate.model_checkpoint='/tmp'",
                "data='debug'",
                "data.struct_template='/weka/scratch/weka/kellislab/rcalef/data/pdb_alphafolddb/AF-%s-F1-model_v4.pdb'",
            ],
        )
    config = instantiate(cfg)
    cross_sims, self_substruct_sims, self_eval_sims = run_one_analysis(
        config=config,
        eval_task=eval_task,
        tot_iters=tot_iters,
        num_workers=num_workers,
    )
    results = {
        "cross_sims": cross_sims,
        "self_substruct_sims": self_substruct_sims,
        "self_eval_sims": self_eval_sims,
    }
    torch.save(results, output_path)


if __name__ == "__main__":
    fire.Fire(run)
