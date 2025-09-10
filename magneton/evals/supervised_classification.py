from pathlib import Path
import json

import torch
import lightning as L

from lightning.pytorch.callbacks import (
    ModelCheckpoint,
)
from lightning.pytorch.loggers import (
    WandbLogger,
)
from torchdata.nodes import Loader
from torchmetrics import (
    Accuracy,
    AveragePrecision,
    MeanAbsoluteError,
    MeanSquaredError,
    Metric,
    MetricCollection,
    SpearmanCorrCoef,
)
from torchmetrics.functional.classification import multilabel_average_precision

from magneton.config import PipelineConfig
from magneton.data import SupervisedDownstreamTaskDataModule, TASK_TYPE
from magneton.data import SupervisedDownstreamTaskDataModule

from .downstream_classifiers import MultiLabelMLP, ResidueClassifier
from .metrics import _calc_fmax

def run_final_predictions(
    model: MultiLabelMLP,
    trainer: L.Trainer,
    loader: Loader,
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

    final_fmax = _calc_fmax(all_logits, all_labels)
    final_auprc = multilabel_average_precision(
        all_logits,
        all_labels,
        num_labels=num_classes,
    )
    print(f"{prefix} final Fmax: {final_fmax.item():0.3f}")
    print(f"{prefix} final AUPRC: {final_auprc.item():0.3f}")


def run_supervised_classification(
    config: PipelineConfig,
    task: str,
    output_dir: str,
    run_id: str,
):
    module = SupervisedDownstreamTaskDataModule(
        data_config=config.data,
        task=task,
        data_dir=config.evaluate.data_dir,
        model_type=config.embedding.model,
        distributed=False,
    )


    # Determine task type based on the task name
    task_type = "multilabel"  # Default for non-PEER tasks
    # PEER multiclass classification tasks
    if task in ["fold"]:
        task_type = "multiclass"
    # PEER binary classification tasks
    elif task in ["solubility", "binary_localization"]:
        task_type = "binary"
    # PEER regression tasks (single sequence)
    elif task in ["fluorescence", "stability", "beta_lactamase", "aav", "gb1", "thermostability", "subcellular_localization"]:
        task_type = "regression"

    # Set up callbacks with appropriate monitoring metric
    if task_type == "multiclass":
        monitor_metric = "valid_accuracy"
        mode = "max"
        filename = "{epoch}-{valid_accuracy:.2f}"
    elif task_type == "binary":
        monitor_metric = "valid_accuracy"
        mode = "max"
        filename = "{epoch}-{valid_auprc:.2f}"
    elif task_type == "regression":
        monitor_metric = "valid_spearman"
        mode = "max"
        filename = "{epoch}-{valid_spearman:.2f}"
    else:  # multilabel
        monitor_metric = "valid_auprc"
        mode = "max"
        filename = "{epoch}-{valid_auprc:.2f}"


    # Get appropriate classifier based on whether this is a protein-level
    # or a residue-level task.
    if module.task_type == TASK_TYPE.PROTEIN_CLASSIFICATION:
        classifier = MultiLabelMLP(
            config=config,
            task=task,
            num_classes=module.num_classes(),
            task_type=task_type,
        )
    elif module.task_type == TASK_TYPE.RESIDUE_CLASSIFICATION:
        classifier = ResidueClassifier(
            config=config,
            task=task,
            num_classes=module.num_classes(),
        )
    else:
        raise ValueError(f"unknown task type: {module.task_type}")


    output_dir = Path(output_dir)
    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir,
            monitor=monitor_metric,
            mode=mode,
            save_top_k=3,
            filename=filename
        ),
        # EarlyStopping(
        #     monitor=monitor_metric,
        #     mode=mode,
        #     patience=4,
        # ),
    ]
    if config.training.dev_run is not None:
        logger = None
        dev_run = config.training.dev_run
    else:
        logger = WandbLogger(
            entity="magneton",
            project="magneton",
            name=run_id,
        )
        dev_run=False

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
    )

    want_distributed_sampler = torch.cuda.device_count() > 1
    module = SupervisedDownstreamTaskDataModule(
        data_config=config.data,
        task=task,
        data_dir=config.evaluate.data_dir,
        model_type=config.embedding.model,
        distributed=want_distributed_sampler,
    )
    if module.task_type == TASK_TYPE.PROTEIN_CLASSIFICATION:
        classifier = MultiLabelMLP(
            config=config,
            task=task,
            num_classes=module.num_classes(),
        )
    elif module.task_type == TASK_TYPE.RESIDUE_CLASSIFICATION:
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
        num_classes=module.num_classes(),
        output_dir=output_dir,
        prefix="validation",
    )

    # Run final predictions
    final_predictions = trainer.predict(
        model=classifier,
        dataloaders=module.val_dataloader(),
        return_predictions=True
    )

    all_logits, all_labels = zip(*final_predictions)
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    torch.save(all_logits, output_dir / "val_logits.pt")
    torch.save(all_labels, output_dir / "val_labels.pt")

    # Calculate task-specific metrics and prepare for JSON export
    metrics_dict = {
        "task": task,
        "task_type": task_type,
        "run_id": run_id,
    }

    if task_type == "multilabel":
        final_fmax = _calc_fmax(all_logits, all_labels)
        final_auprc = multilabel_average_precision(
            all_logits,
            all_labels,
            num_labels=module.num_classes(),
        )
        print(f"Final Fmax: {final_fmax.item():0.3f}")
        print(f"Final AUPRC: {final_auprc.item():0.3f}")
        metrics_dict.update({
            "fmax": float(final_fmax.item()),
            "auprc": float(final_auprc.item()),
        })
    elif task_type == "multiclass":
        # For multiclass, calculate accuracy
        predicted_classes = all_logits.argmax(dim=1)
        accuracy = (predicted_classes == all_labels).float().mean()
        print(f"Final Accuracy: {accuracy.item():0.3f}")
        metrics_dict.update({
            "accuracy": float(accuracy.item()),
        })
    elif task_type == "binary":
        # For binary classification
        predicted_probs = all_logits.sigmoid()
        predicted_classes = (predicted_probs >= 0.5).float()
        accuracy = (predicted_classes == all_labels).float().mean()
        auprc = AveragePrecision(task="binary")(all_logits, all_labels.long())
        print(f"Final Accuracy: {accuracy.item():0.3f}")
        print(f"Final AUPRC: {auprc.item():0.3f}")
        metrics_dict.update({
            "accuracy": float(accuracy.item()),
            "auprc": float(auprc.item()),
        })
    elif task_type == "regression":
        # For regression tasks - ensure shape compatibility
        if all_labels.dim() == 1:
            all_labels = all_labels.unsqueeze(-1)  # Match logits shape [N, 1]

        mae = MeanAbsoluteError()(all_logits, all_labels)
        rmse = MeanSquaredError(squared=False)(all_logits, all_labels)
        spearman = SpearmanCorrCoef()(all_logits.squeeze(), all_labels.squeeze())
        print(f"Final MAE: {mae.item():0.3f}")
        print(f"Final RMSE: {rmse.item():0.3f}")
        print(f"Final Spearman: {spearman.item():0.3f}")
        metrics_dict.update({
            "mae": float(mae.item()),
            "rmse": float(rmse.item()),
            "spearman": float(spearman.item()),
        })
    else:
        print(f"No final metrics defined for task_type: {task_type}")
        metrics_dict["error"] = f"No metrics defined for task_type: {task_type}"

    # Save metrics to JSON file
    metrics_json_path = Path(output_dir) / f"{task}_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Metrics saved to: {metrics_json_path}")

    if logger is not None:
        logger.experiment.finish()
    run_final_predictions(
        model=classifier,
        trainer=trainer,
        loader=module.test_dataloader(),
        num_classes=module.num_classes(),
        output_dir=output_dir,
        prefix="test",
    )
