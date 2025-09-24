import json
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
from magneton.training.embedding_mlp import EmbeddingMLP, MultitaskEmbeddingMLP


def classify_substructs(
    config: PipelineConfig,
    output_dir: Path,
):
    if config.data.collapse_labels:
        model = EmbeddingMLP.load_from_checkpoint(
            config.evaluate.model_checkpoint,
        )
    else:
        model = MultitaskEmbeddingMLP.load_from_checkpoint(
            config.evaluate.model_checkpoint,
        )
    model.eval()

    data_module = MagnetonDataModule(
        data_config=config.data,
        model_type=config.embedding.model,
    )

    substruct_parser = get_substructure_parser(config.data)
    num_classes=substruct_parser.num_labels()

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

    all_losses, all_logits, all_labels = zip(*final_predictions)
    all_losses = torch.stack(all_losses)
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

    torch.save(results_dict, output_dir / "test_results.pt")

    metrics = MetricCollection({
        "macro_accuracy": Accuracy(
            task="multiclass", average="macro", num_classes=num_classes
        ),
        "macro_auprc": AveragePrecision(
            task="multiclass", average="macro", num_classes=num_classes
        ),
        "macro_auroc": AUROC(
            task="multiclass", average="macro", num_classes=num_classes
        ),
    })

    metrics_dict = {
        "task": "substructure",
    }
    metrics_dict.update(
        {k: v.item() for k, v in metrics(all_logits, all_labels).items()}
    )

    print(f"final metrics: {metrics_dict}")

    # Save metrics to JSON file
    metrics_json_path = Path(output_dir) / "test_substructure_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Metrics saved to: {metrics_json_path}")
