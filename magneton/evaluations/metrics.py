import math

import numpy as np
import torch
from torchmetrics import (
    AUROC,
    Accuracy,
    AveragePrecision,
    MeanAbsoluteError,
    MeanSquaredError,
    Metric,
    MetricCollection,
    SpearmanCorrCoef,
)

from magneton.data.evals import EVAL_TASK


def format_logits_and_labels_for_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    task_type: EVAL_TASK,
) -> torch.Tensor:
    """Perform any reshaping and dtype conversions for metric calculations."""
    if task_type == EVAL_TASK.BINARY:
        # For binary metrics, convert labels to int (AveragePrecision expects int targets)
        labels = labels.int()
        #if logits.shape[0] != 1:
        print(logits.shape)
        print(logits.squeeze(-1).shape)
        return logits.squeeze(-1), labels.squeeze(-1)
    elif task_type in [EVAL_TASK.MULTILABEL, EVAL_TASK.MULTICLASS]:
        return logits, labels.int()
    else:
        return logits, labels


def get_task_torchmetrics(
    task_type: EVAL_TASK,
    num_classes: int,
    prefix: str,
) -> MetricCollection:
    """Get the metrics used for a given task type."""
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
            "auroc": AUROC(task="binary"),
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


def _calc_fmax(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_thresh_steps: int = 101,
) -> torch.Tensor:
    """Compute Fmax score, i.e. maximum F1 over possible thresholds.

    Args:
        - logits (torch.Tensor): class logits
        - labels (torch.Tensor): one-hot encoded labels
        - num_thresh_steps: number of steps in [0, 1] to consider
            as score threshold for positive call
    """
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
    """Metric object for Fmax score, i.e. maximum F1 over possible thresholds."""
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


class PrecisionAtL(Metric):
    """Compute precision at L (P@L) for short, medium, and long-range contacts.

    Precision at L is defined as the number of true contacts in the L-most confident
    contact predictions for a protein at length L. This metric computes P@L, P@L/2, and P@L/5
    (i.e. the L/2 and L/5 most confident contact predictions, respectively). For each P@L,
    we consider short-, medium-, and long-range contacts defined by considering residue
    pairs whose positions in the primary sequence are separated by [6, 11], [12, 23], and [24, L]
    respectively.

    Credit for metric calculation code goes to the SaProt authors:
      https://github.com/westlake-repl/SaProt/blob/main/model/saprot/saprot_contact_model.py#L81
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.lengths = {"P@L": 1, "P@L/2": 2, "P@L/5": 5}
        self.ranges = ["short_range", "medium_range", "long_range"]

        accuracies = {}
        for length in self.lengths.keys():
            for range in self.ranges:
                accuracies[f"{range}_{length}"] = Accuracy(task="binary", ignore_index=-1, **kwargs)
        self.accuracies = MetricCollection(accuracies)

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        protein_lengths: list[int],
    ) -> None:
        if preds.shape != targets.shape:
            raise ValueError(f"preds and target must have the same shape: {preds.shape} != {targets.shape}")

        for pred_map, label_map, L in zip(preds, targets, protein_lengths):
            x_inds, y_inds = np.indices(label_map.shape)
            for r in self.ranges:
                if r == "short_range":
                    mask = (np.abs(y_inds - x_inds) < 6) | (
                        np.abs(y_inds - x_inds) > 11
                    )

                elif r == "medium_range":
                    mask = (np.abs(y_inds - x_inds) < 12) | (
                        np.abs(y_inds - x_inds) > 23
                    )

                else:
                    mask = np.abs(y_inds - x_inds) < 24

                mask = torch.from_numpy(mask)
                copy_label_map = label_map.clone()
                copy_label_map[mask] = -1

                # Mask the lower triangle
                mask = torch.triu(torch.ones_like(copy_label_map), diagonal=1)
                copy_label_map[mask == 0] = -1

                selector = copy_label_map != -1
                probs = pred_map[selector].float()
                labels = copy_label_map[selector]

                for k, v in self.lengths.items():
                    l = min(math.ceil(L / v), (labels == 1).sum().item())

                    top_inds = torch.argsort(probs, descending=True)[:l]
                    top_labels = labels[top_inds]

                    if top_labels.numel() == 0:
                        continue

                    metric = f"{r}_{k}"
                    self.accuracies[metric].update(
                        top_labels, torch.ones_like(top_labels)
                    )

    def compute(self) -> dict[str, torch.Tensor]:
        return self.accuracies.compute()
