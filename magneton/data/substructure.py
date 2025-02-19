import os
from dataclasses import dataclass
from typing import List

import pandas as pd
import torch

from magneton.types import InterProType, Protein


@dataclass
class LabeledSubstructure:
    ranges: List[torch.Tensor]
    label: int

    def to(self, device: str):
        for i in range(len(self.ranges)):
            self.ranges[i] = self.ranges[i].to(device)
        return self


class SubstructureParser:
    def __init__(
        self,
        want_types: List[InterProType],
        labels_dir: str,
    ):
        self.want_types = sorted(want_types)
        self.type_to_label = {}

        curr_label = 0
        for type in self.want_types:
            labels = pd.read_table(os.path.join(labels_dir, f"{type}.labels.tsv"))
            labels.label += curr_label
            for label, interpro_id in labels[["label", "interpro_id"]].itertuples(index=False):
                self.type_to_label[interpro_id] = label

            curr_label += len(labels)

    def parse(self, prot: Protein) -> List[LabeledSubstructure]:
        """
        Parse the protein and return a tensor of ranges and labels.
        """
        parsed = []
        for entry in prot.entries:
            if entry.id not in self.type_to_label:
                continue
            if entry.element_type in [InterProType.DOMAIN, InterProType.FAMILY] and not entry.representative:
                continue
            parsed.append(LabeledSubstructure(
                ranges=[torch.tensor((start, end)) for start, end in entry.positions],
                label=self.type_to_label[entry.id],
            ))
        return parsed