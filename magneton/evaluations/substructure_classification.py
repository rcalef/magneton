import json
from collections import defaultdict
from pathlib import Path

import lightning as L
import torch
from torchmetrics import (
    AUROC,
    Accuracy,
    AveragePrecision,
    MetricCollection,
)

from magneton.config import PipelineConfig
from magneton.data import MagnetonDataModule
from magneton.data.core import get_substructure_parser
from magneton.models.substructure_classifier import SubstructureClassifier


def classify_substructs(
    config: PipelineConfig,
    output_dir: Path,
):
    model = SubstructureClassifier.load_from_checkpoint(
        config.evaluate.model_checkpoint,
    )

    data_module = MagnetonDataModule(
        data_config=config.data,
        model_type=config.embedding.model,
    )

    substruct_parser = get_substructure_parser(config.data)
    num_classes = substruct_parser.num_labels()

    trainer = L.Trainer(
        accelerator=config.training.accelerator,
        devices=config.training.devices,
        default_root_dir=output_dir,
        precision=config.training.precision,
        use_distributed_sampler=False,
        fast_dev_run=config.training.dev_run,
        logger=False,
    )
    loader = data_module.test_dataloader()

    final_predictions = trainer.predict(
        model=model, dataloaders=loader, return_predictions=True
    )
    
    logits_by_task = defaultdict(list)
    labels_by_task = defaultdict(list)
    for logits_dict, labels_dict in final_predictions:
        for substruct_type, logits in logits_dict.items():
            logits_by_task[substruct_type].append(logits)
        for substruct_type, labels in labels_dict.items():
            labels_by_task[substruct_type].append(labels)

    # Concatenate all batches
    for substruct_type in logits_by_task.keys():
        logits_by_task[substruct_type] = torch.cat(logits_by_task[substruct_type])
        labels_by_task[substruct_type] = torch.cat(labels_by_task[substruct_type])

    metrics_dict = {"task": "multitask_substructure"}

    for substruct_type, logits in logits_by_task.items():
        this_num_classes = num_classes[substruct_type]

        metrics = MetricCollection({
            "macro_accuracy": Accuracy(
                task="multiclass", average="macro", num_classes=this_num_classes
            ),
            "macro_auprc": AveragePrecision(
                task="multiclass", average="macro", num_classes=this_num_classes
            ),
            "macro_auroc": AUROC(
                task="multiclass", average="macro", num_classes=this_num_classes
            ),
        })

        task_metrics = metrics(logits, labels_by_task[substruct_type])
        for k, v in task_metrics.items():
            metrics_dict[f"{substruct_type}_{k}"] = v.item()

    print(f"final metrics: {metrics_dict}")

    # Save metrics to JSON
    metrics_json_path = Path(output_dir) / "test_substructure_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Metrics saved to: {metrics_json_path}")