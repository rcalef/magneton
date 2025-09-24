import json
from pathlib import Path

import lightning as L
import torch
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

from .downstream_classifiers import (
    ContactPredictor,
    MultiLabelMLP,
    PPIPredictor,
    ResidueClassifier,
)
from .metrics import (
    FMaxScore,
    PrecisionAtL,
    format_logits_and_labels_for_metrics,
    get_task_torchmetrics,
)


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
        model=model, dataloaders=loader, return_predictions=True
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
    if task_type == EVAL_TASK.MULTILABEL:
        # Fmax isn't included by default since it's too expensive to compute
        # as a metric on every iteration
        metrics.add_metrics({"fmax": FMaxScore(num_thresh_steps=101)})

    metrics_dict = {
        "task": task,
        "task_type": task_type,
    }

    all_logits, all_labels = format_logits_and_labels_for_metrics(
        all_logits,
        all_labels,
        task_type,
    )
    metrics_dict.update(
        {k: v.item() for k, v in metrics(all_logits, all_labels).items()}
    )

    print(f"{prefix} final metrics: {metrics_dict}")

    # Save metrics to JSON file
    metrics_json_path = Path(output_dir) / f"{prefix}_{task}_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Metrics saved to: {metrics_json_path}")


def run_final_contact_predictions(
    model: ContactPredictor,
    trainer: L.Trainer,
    loader: Loader,
    output_dir: Path,
    prefix: str,
):
    final_predictions = trainer.predict(
        model=model, dataloaders=loader, return_predictions=True
    )

    # Make another pass through dataset to collect labels, IDs, and lengths
    all_labels = []
    protein_ids = []
    protein_lengths = []
    for batch in loader:
        all_labels.append(batch.labels)
        protein_ids.append(batch.protein_ids)
        protein_lengths.append(batch.lengths)

    results_dict = {
        "protein_ids": protein_ids,
        "lengths": protein_lengths,
        "logits": final_predictions,
        "labels": all_labels,
    }

    torch.save(results_dict, output_dir / f"{prefix}_results.pt")

    # Calculate task-specific metrics and prepare for JSON export
    metrics_dict = {
        "task": "contact_prediction",
    }
    p_at_l = PrecisionAtL(sync_on_compute=False)
    for logits, labels, lengths in zip(final_predictions, all_labels, protein_lengths):
        p_at_l.update(logits, labels, lengths)

    metrics_dict.update({k: v.item() for k, v in p_at_l.compute().items()})

    print(f"{prefix} final metrics: {metrics_dict}")

    # Save metrics to JSON file
    metrics_json_path = Path(output_dir) / f"{prefix}_contact_prediction_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Metrics saved to: {metrics_json_path}")


def run_supervised_classification(
    config: PipelineConfig,
    task: str,
    output_dir: Path,
    run_id: str,
):
    if task not in TASK_TO_TYPE:
        raise ValueError(f"unknown task: {task}")
    task_type = TASK_TO_TYPE[task]

    expected_final_metrics_path = output_dir / f"test_{task}_metrics.json"
    if expected_final_metrics_path.exists():
        if not config.evaluate.rerun_completed:
            print(f"{task}: final predictions preset, recompute=False, skipping")
            return
        else:
            print(f"{task}: final predictions preset, recompute=True, rerunning")

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
            filename=filename,
        ),
    ]
    is_dev_run = (type(config.training.dev_run) is int) or bool(config.training.dev_run)
    if is_dev_run or config.evaluate.final_prediction_only:
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
        unk_amino_acid_char=config.embedding.model_params.get("unk_amino_acid_char", "X"),
    )

    if module.task_granularity == TASK_GRANULARITY.PROTEIN_CLASSIFICATION:
        classifier_cls = MultiLabelMLP
    elif module.task_granularity == TASK_GRANULARITY.RESIDUE_CLASSIFICATION:
        classifier_cls = ResidueClassifier
    elif module.task_granularity == TASK_GRANULARITY.CONTACT_PREDICTION:
        classifier_cls = ContactPredictor
    elif module.task_granularity == TASK_GRANULARITY.PPI_PREDICTION:
        classifier_cls = PPIPredictor
    else:
        raise ValueError(f"unknown task type: {module.task_type}")

    # Either train a new downstream classifier head, or load an existing checkpoint
    final_ckpt_path = output_dir / "final_model.ckpt"
    if not config.evaluate.final_prediction_only:
        classifier = classifier_cls(
            config=config,
            task=task,
            num_classes=module.num_classes(),
            task_type=task_type,
        )

        trainer.fit(
            classifier,
            datamodule=module,
        )
        # Always save a final checkpoint
        trainer.save_checkpoint(final_ckpt_path)
    else:
        if not final_ckpt_path.exists():
            raise FileNotFoundError(f"No final checkpoint fount at: {final_ckpt_path}")
        print(f"{task}: running predictions preset with model at: {final_ckpt_path}")

        classifier = classifier_cls.load_from_checkpoint(
            final_ckpt_path,
        )

    # Disable distributed sampler for final validations for reproducibility
    module.distributed = False
    if task == "contact_prediction":
        run_final_contact_predictions(
            model=classifier,
            trainer=trainer,
            loader=module.val_dataloader(),
            output_dir=output_dir,
            prefix="validation",
        )

        run_final_contact_predictions(
            model=classifier,
            trainer=trainer,
            loader=module.test_dataloader(),
            output_dir=output_dir,
            prefix="test",
        )
    else:
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
