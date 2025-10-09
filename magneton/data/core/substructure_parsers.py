import os
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
import torch

from magneton.config import DataConfig
from magneton.core_types import (
    DSSP_TO_NAME,
    INTERPRO_REP_TYPES,
    Protein,
    SubstructType,
)


@dataclass
class LabeledSubstructure:
    ranges: list[torch.Tensor]
    label: int
    element_type: SubstructType

    def to(self, device: str):
        for i in range(len(self.ranges)):
            self.ranges[i] = self.ranges[i].to(device)
        return self


class BaseSubstructureParser(ABC):
    def parse(self, prot: Protein) -> list[LabeledSubstructure]:
        """
        Parse the protein and return a tensor of ranges and labels.
        """
        raise NotImplementedError("Must be implemented in subclass")


class UnifiedSubstructureParser(BaseSubstructureParser):
    """Converts substructures to labels and ranges. All substructures share a label set.

    Takes in a `Protein` object and returns a list of `LabeledSubstructure` objects,
    subsetting to substructures of the desired types, i.e. those specified in `want_types`
    at init time.

    All InterPro types listed in `want_types` are collapsed into a single unified label set,
    e.g. if `want_types` contains `Conserved_Site` (N = 50) and  `Domain` (N = 3000), then
    the resulting parser will generate labels in [0, 3050), with the first 50 labels
    corresponding to conserved sites and the last 3000 labels corresponding to domains.
    """

    def __init__(
        self,
        want_types: list[SubstructType],
        labels_dir: str,
        elem_name: str = "all",
    ):
        self.want_types = sorted(want_types)
        self.elem_name = elem_name
        self.type_to_label = {}
        self.parse_ss = False

        curr_label = 0
        for type in self.want_types:
            if type == SubstructType.SS:
                labels = [(i + curr_label, name) for i, name in enumerate(DSSP_TO_NAME)]
                self.parse_ss = True
            else:
                labels = pd.read_table(os.path.join(labels_dir, f"{type}.labels.tsv"))
                labels.label += curr_label
                labels = list(labels[["label", "interpro_id"]].itertuples(index=False))

            for label, name in labels:
                self.type_to_label[name] = label

            curr_label += len(labels)

    def parse(self, prot: Protein) -> list[LabeledSubstructure]:
        """
        Parse the protein and return a tensor of ranges and labels.
        """
        parsed = []
        for entry in prot.entries:
            if entry.id not in self.type_to_label:
                continue
            if entry.element_type in INTERPRO_REP_TYPES and not entry.representative:
                continue
            parsed.append(
                LabeledSubstructure(
                    ranges=[
                        torch.tensor((start, end)) for start, end in entry.positions
                    ],
                    label=self.type_to_label[entry.id],
                    element_type=self.elem_name,
                )
            )

        if self.parse_ss:
            for ss in prot.secondary_structs:
                parsed.append(
                    LabeledSubstructure(
                        ranges=[torch.tensor((ss.start, ss.end))],
                        label=self.type_to_label[DSSP_TO_NAME[ss.dssp_type]],
                        element_type=SubstructType.SS,
                    )
                )
        return parsed

    def num_labels(self) -> int:
        return len(self.type_to_label)


class SeparatedSubstructureParser(BaseSubstructureParser):
    """Converts substructures to labels and ranges. Each substructure type has its own label set.

    Same as `UnifiedSubstructureParse` above, except substructure types DO NOT share a label set,
    e.g. if `want_types` contains `Domain` (N = 3000) and `Conserved_Site` (N = 50), then the
    resulting parser will generate labels in [0, 3000) for domains and [0, 50) for conserved
    sites.
    """

    def __init__(
        self,
        want_types: list[SubstructType],
        labels_dir: str,
    ):
        self.want_types = sorted(want_types)
        self.type_to_label = defaultdict(dict)
        self.parse_ss = False

        for type in self.want_types:
            if type == SubstructType.SS:
                self.parse_ss = True
                labels = list(enumerate(DSSP_TO_NAME))
            else:
                labels = pd.read_table(os.path.join(labels_dir, f"{type}.labels.tsv"))
                labels = list(labels[["label", "interpro_id"]].itertuples(index=False))

            for label, name in labels:
                self.type_to_label[type][name] = label

    def parse(self, prot: Protein) -> list[LabeledSubstructure]:
        """
        Parse the protein and return a tensor of ranges and labels.
        """
        parsed = []
        for entry in prot.entries:
            if entry.element_type not in self.type_to_label:
                continue
            if entry.id not in self.type_to_label[entry.element_type]:
                continue
            if entry.element_type in INTERPRO_REP_TYPES and not entry.representative:
                continue
            parsed.append(
                LabeledSubstructure(
                    ranges=[
                        torch.tensor((start, end)) for start, end in entry.positions
                    ],
                    label=self.type_to_label[entry.element_type][entry.id],
                    element_type=entry.element_type,
                )
            )
        if self.parse_ss:
            for ss in prot.secondary_structs:
                parsed.append(
                    LabeledSubstructure(
                        ranges=[torch.tensor((ss.start, ss.end))],
                        label=ss.dssp_type,
                        element_type=SubstructType.SS,
                    )
                )

        return parsed

    def num_labels(self) -> dict[str, int]:
        return {type: len(labels) for type, labels in self.type_to_label.items()}


def get_substructure_parser(data_config: DataConfig) -> BaseSubstructureParser:
    if data_config.collapse_labels:
        # TODO: write out what this unified label set actually is
        return UnifiedSubstructureParser(
            want_types=data_config.substruct_types,
            labels_dir=data_config.labels_path,
            elem_name="all"
            if len(data_config.substruct_types) > 1
            else data_config.substruct_types[0],
        )
    else:
        return SeparatedSubstructureParser(
            want_types=data_config.substruct_types,
            labels_dir=data_config.labels_path,
        )
