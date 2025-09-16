import torch
import torch.distributed as dist
from torchmetrics import (
    Accuracy,
    AveragePrecision,
    MeanAbsoluteError,
    MeanSquaredError,
    Metric,
    MetricCollection,
    SpearmanCorrCoef,
)

from magneton.data.evals import EVAL_TASK


def _calc_fmax(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_thresh_steps: int = 101,
) -> torch.Tensor:
    assert ((labels == 0) | (labels == 1)).all()

    probs = logits.sigmoid()
    f1s = []
    for thresh in torch.linspace(start=0, end=1, steps=num_thresh_steps):
        preds = probs >= thresh

        tp = ((preds == labels) & labels).sum()
        fp = ((preds != labels) & labels).sum()
        fn = ((preds != labels) & ~labels).sum()

        f1 = (2 * tp) / (2 * tp + fp + fn)
        f1s.append(f1)

    f1s = torch.stack(f1s)
    return f1s.max()


class FMaxScore(Metric):
    def __init__(self, **kwargs):
        self.num_thresh_steps = kwargs.pop("num_thresh_steps", 101)
        super().__init__(**kwargs)

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("labels", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        # Revisit this if it becomes a memory issue. Previously was
        # moving these to CPU, but this would require also handling
        # a gloo backend when running with more than one GPU.
        self.preds.append(preds.detach())
        self.labels.append(target.detach())

    def compute(self) -> torch.Tensor:
        # Probably a better way to handle this, but the list of
        # tensors is already torch.cat'd if running in distributed
        # setting.
        if isinstance(self.preds, list):
            all_preds = torch.cat(self.preds)
            all_labels = torch.cat(self.labels)
        else:
            all_preds = self.preds
            all_labels = self.labels

        return _calc_fmax(all_preds, all_labels)


def get_task_torchmetrics(
    task_type: EVAL_TASK,
    num_classes: int,
    prefix: str,
) -> MetricCollection:
    if task_type == EVAL_TASK.MULTILABEL:
        metrics = {
            "accuracy": Accuracy(task="multilabel", num_labels=num_classes),
            "auprc": AveragePrecision(task="multilabel", num_labels=num_classes),
        }
    elif task_type == EVAL_TASK.MULTICLASS:
        metrics = {
            "accuracy": Accuracy(task="multiclass", num_classes=num_classes),
        }
    elif task_type == EVAL_TASK.BINARY:
        metrics = {
            "accuracy": Accuracy(task="binary"),
            "auprc": AveragePrecision(task="binary"),
        }
    elif task_type == EVAL_TASK.REGRESSION:
        metrics = {
            "mae": MeanAbsoluteError(),
            "rmse": MeanSquaredError(squared=False),  # RMSE instead of MSE
            "spearman": SpearmanCorrCoef(),
        }
    else:
        raise ValueError(f"unknown task type: {task_type}")
    return MetricCollection(metrics, prefix=prefix)


def format_logits_and_labels_for_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    task_type: EVAL_TASK,
) -> torch.Tensor:
    if task_type == EVAL_TASK.BINARY:
        # For binary metrics, convert labels to int (AveragePrecision expects int targets)
        return logits.squeeze(), labels.squeeze().long()
    else:
        return logits, labels
