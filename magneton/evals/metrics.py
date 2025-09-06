import torch
from torchmetrics import (
    Metric,
)

def _calc_fmax(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_thresh_steps: int=101,
) -> torch.Tensor:
    probs = logits.sigmoid()
    f1s = []
    for thresh in torch.linspace(start=0, end=1, steps=num_thresh_steps):
        preds = probs >= thresh

        tp = ((preds == labels) & labels).sum()
        fp = ((preds != labels) & labels).sum()
        tn = ((preds == labels) & ~labels).sum()
        fn = ((preds != labels) & ~labels).sum()

        f1 = (2*tp) / (2*tp + fp + fn)
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

        self.preds.append(preds.detach().cpu())
        self.labels.append(target.detach().cpu())

    def compute(self) -> torch.Tensor:
        all_preds = torch.cat(self.preds)
        all_labels = torch.cat(self.labels)

        return _calc_fmax(all_preds, all_labels)