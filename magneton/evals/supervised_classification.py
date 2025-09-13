import json

from pathlib import Path

import torch
import lightning as L

from lightning.pytorch.callbacks import (
    ModelCheckpoint,
)
from lightning.pytorch.loggers import (
    WandbLogger,
)
from torchdata.nodes import Loader

from magneton.config import PipelineConfig
from magneton.data import (
    SupervisedDownstreamTaskDataModule,
)
from magneton.data.evals import (
    EVAL_TASK,
    TASK_GRANULARITY,
    TASK_TO_TYPE,
)

from .metrics import format_logits_and_labels_for_metrics, get_task_torchmetrics
from .downstream_classifiers import MultiLabelMLP, ResidueClassifier

def run_final_predictions(
    model: MultiLabelMLP,
    trainer: L.Trainer,
    loader: Loader,
    task: str,
    num_classes: int,
    output_dir: Path,
    prefix: str,
):
    final_predictions = trainer.predict(
        model=model,
        dataloaders=loader,
        return_predictions=True
    )

    all_logits, all_labels = zip(*final_predictions)
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)


    # Make another pass through dataset to collect IDs
    protein_ids = []
    for batch in loader:
        protein_ids.extend(batch.protein_ids)

    results_dict = {
        "protein_ids": protein_ids,
        "logits": all_logits,
        "labels": all_labels,
    }

    torch.save(results_dict, output_dir / f"{prefix}_results.pt")

    # Calculate task-specific metrics and prepare for JSON export
    task_type = TASK_TO_TYPE[task]
    metrics = get_task_torchmetrics(task_type, num_classes, prefix=f"{prefix}_")
    metrics_dict = {
        "task": task,
        "task_type": task_type,
    }

    all_logits, all_labels = format_logits_and_labels_for_metrics(
        all_logits,
        all_labels,
        task_type,
    )
    metrics_dict.update({k: v.item() for k,v in metrics(all_logits, all_labels).items()})

    print(f"{prefix} final metrics: {metrics_dict}")

    # Save metrics to JSON file
    metrics_json_path = Path(output_dir) / f"{prefix}_{task}_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Metrics saved to: {metrics_json_path}")

def run_supervised_classification(
    config: PipelineConfig,
    task: str,
    output_dir: str,
    run_id: str,
):
    if task not in TASK_TO_TYPE:
        raise ValueError(f"unknown task: {task}")
    task_type = TASK_TO_TYPE[task]

    output_dir = Path(output_dir)
    # Set up callbacks with appropriate monitoring metric
    if task_type == EVAL_TASK.MULTICLASS:
        monitor_metric = "valid_accuracy"
        filename = "{epoch}-{valid_accuracy:.2f}"
    elif task_type == EVAL_TASK.BINARY:
        monitor_metric = "valid_accuracy"
        filename = "{epoch}-{valid_auprc:.2f}"
    elif task_type == EVAL_TASK.REGRESSION:
        monitor_metric = "valid_spearman"
        filename = "{epoch}-{valid_spearman:.2f}"
    elif task_type == EVAL_TASK.MULTILABEL:  # multilabel
        monitor_metric = "valid_auprc"
        filename = "{epoch}-{valid_auprc:.2f}"

    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir,
            monitor=monitor_metric,
            mode="max",
            save_top_k=3,
            filename=filename
        ),
    ]
    is_dev_run = config.training.dev_run or isinstance(config.training.dev_run, int)
    if is_dev_run:
        logger = None
    else:
        logger = WandbLogger(
            entity="magneton",
            project="magneton",
            name=run_id,
        )

    # Create trainer
    trainer = L.Trainer(
        strategy=config.training.strategy,
        callbacks=callbacks,
        logger=logger,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
        default_root_dir=output_dir,
        max_epochs=config.training.max_epochs,
        precision=config.training.precision,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        val_check_interval=1.0,
        use_distributed_sampler=False,
        fast_dev_run=config.training.dev_run,
    )

    want_distributed_sampler = torch.cuda.device_count() > 1
    module = SupervisedDownstreamTaskDataModule(
        data_config=config.data,
        task=task,
        data_dir=config.evaluate.data_dir,
        model_type=config.embedding.model,
        distributed=want_distributed_sampler,
    )

    if module.task_granularity == TASK_GRANULARITY.PROTEIN_CLASSIFICATION:
        classifier = MultiLabelMLP(
            config=config,
            task=task,
            num_classes=module.num_classes(),
            task_type=task_type,
        )
    elif module.task_granularity == TASK_GRANULARITY.RESIDUE_CLASSIFICATION:
        classifier = ResidueClassifier(
            config=config,
            task=task,
            num_classes=module.num_classes(),
        )
    else:
        raise ValueError(f"unknown task type: {module.task_type}")

    trainer.fit(
        classifier,
        datamodule=module,
    )
    # Always save a final checkpoint
    trainer.save_checkpoint(output_dir / "final_model.ckpt")

    # Disable distributed sampler for final validations for reproducibility
    module.distributed = False
    run_final_predictions(
        model=classifier,
        trainer=trainer,
        loader=module.val_dataloader(),
        task=task,
        num_classes=module.num_classes(),
        output_dir=output_dir,
        prefix="validation",
    )

    run_final_predictions(
        model=classifier,
        trainer=trainer,
        loader=module.test_dataloader(),
        task=task,
        num_classes=module.num_classes(),
        output_dir=output_dir,
        prefix="test",
    )

    if logger is not None:
        logger.experiment.finish()
