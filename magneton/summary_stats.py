import os

from collections import defaultdict
from dataclasses import dataclass
from typing import (
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
)

import pandas as pd

from magneton.custom_types import (
    InterproEntry,
    Protein,
    SecondaryStructure,
    DSSP_TO_NAME,
)

from magneton.custom_types import DSSP_TO_NAME

want_types = [
    "Family",
    "Domain",
    "Homologous_superfamily",
    "Conserved_site",
    "Repeat",
    "Active_site",
    "Binding_site",
    "PTM",
]

interpro_cat_dtype = pd.CategoricalDtype(categories=want_types, ordered=True)

# These InterPro element types actually use the `representative` field.
have_representative = {
    "Family",
    "Domain",
    "Repeat",
}

ss_cat_dtype = pd.CategoricalDtype(categories=DSSP_TO_NAME, ordered=True)


def increment_average(
    old_avg: Optional[float],
    new_val: float,
    count: int,
) -> float:
    """Update a running average"""
    if old_avg is None:
        return new_val
    return old_avg + (new_val - old_avg) / (count + 1)


def combine_averages(
    avg_a: Optional[float],
    count_a: int,
    avg_b: Optional[float],
    count_b: int,
) -> Tuple[float, int]:
    """Combine two running averages"""
    if avg_a is None:
        return avg_b, count_b
    if avg_b is None:
        return avg_a, count_a
    new_count = count_a + count_b
    new_avg = avg_a * (count_a / new_count) + avg_b * (count_b / new_count)
    return new_avg, new_count


def summarize_protein(
    prot: Protein,
) -> pd.DataFrame:
    """Calculate summary statistics about the InterPro entries in a single protein"""
    filtered_entries = [
        x
        for x in prot.entries
        if (x.element_type in have_representative and x.representative == "true")
        or (x.element_type not in have_representative)
    ]
    entry_types = [x.element_type for x in filtered_entries]
    interpro_counts = pd.Series(entry_types, dtype=interpro_cat_dtype).value_counts()

    ss_types = [DSSP_TO_NAME[x.dssp_type.value] for x in prot.secondary_structs]
    ss_counts = pd.Series(ss_types, dtype=ss_cat_dtype).value_counts()
    ss_cov = sum([x.end - x.start for x in prot.secondary_structs]) / int(prot.length)

    # Concat counts into one dict and return as a single-row DataFrame
    ret = {
        "length": prot.length,
        "ss_cov": ss_cov,
    }
    ret.update(interpro_counts.to_dict())
    ret.update(ss_counts.to_dict())

    return pd.DataFrame(ret, index=[prot.uniprot_id])

@dataclass
class SubstructureMetrics:
    element_name: str
    count: int
    max_occurences: int
    min_len: Optional[int]
    max_len: Optional[int]
    avg_len: Optional[float]
    min_num_segs: Optional[int]
    max_num_segs: Optional[int]
    avg_num_segs: Optional[float]
    avg_coverage: Optional[float]

    def __init__(
        self,
        name: str,
        id: str,
    ):
        self.element_id = id
        self.element_name = name
        self.count = 0
        self.max_occurences = 0
        self.min_len = None
        self.max_len = None
        self.avg_len = None
        self.min_num_segs = None
        self.max_num_segs = None
        self.avg_num_segs = None
        self.avg_coverage = None

    def update(
        self,
        length: int,
        num_segs: int,
        prot_length: int,
        num_occurences: int,
    ):
        if self.min_len is None or length < self.min_len:
            self.min_len = length
        if self.max_len is None or length > self.max_len:
            self.max_len = length

        # Update running average of length.
        self.avg_len = increment_average(
            self.avg_len,
            length,
            self.count,
        )

        if self.min_num_segs is None or num_segs < self.min_num_segs:
            self.min_num_segs = num_segs
        if self.max_num_segs is None or num_segs > self.max_num_segs:
            self.max_num_segs = num_segs

        # Update running average of num_segs.
        self.avg_num_segs = increment_average(
            self.avg_num_segs,
            num_segs,
            self.count,
        )

        # Update running average of coverage.
        self.avg_coverage = increment_average(
            self.avg_coverage,
            length / prot_length,
            self.count,
        )

        self.count += 1

        if num_occurences > self.max_occurences:
            self.max_occurences = num_occurences

    def update_from_interpro(
        self,
        entry: InterproEntry,
        prot_length: int,
        num_occurences: int,
    ):
        assert entry.id == self.element_id
        entry_len = sum([end - start for start, end in entry.positions])
        self.update(
            entry_len,
            len(entry.positions),
            prot_length,
            num_occurences,
        )

    def update_from_secondary_struct(
        self,
        struct: SecondaryStructure,
        prot_length: int,
        num_occurences: int,
    ):
        assert struct.dssp_type.value == self.element_id
        entry_len = struct.end - struct.start
        self.update(
            entry_len,
            1,
            prot_length,
            num_occurences,
        )

    def merge(
        self,
        rhs,
    ):
        if rhs.min_len is not None:
            if self.min_len is None or rhs.min_len < self.min_len:
                self.min_len = rhs.min_len
        if rhs.max_len is not None:
            if self.max_len is None or rhs.max_len > self.max_len:
                self.max_len = rhs.max_len
        self.avg_len, _ = combine_averages(
            self.avg_len,
            self.count,
            rhs.avg_len,
            rhs.count,
        )
        if rhs.min_num_segs is not None:
            if self.min_num_segs is None or rhs.min_num_segs < self.min_num_segs:
                self.min_num_segs = rhs.min_num_segs
        if rhs.max_num_segs is not None:
            if self.max_num_segs is None or rhs.max_num_segs > self.max_num_segs:
                self.max_num_segs = rhs.max_num_segs
        self.avg_num_segs, _ = combine_averages(
            self.avg_num_segs,
            self.count,
            rhs.avg_num_segs,
            rhs.count,
        )

        self.avg_coverage, _ = combine_averages(
            self.avg_coverage,
            self.count,
            rhs.avg_coverage,
            rhs.count,
        )

        self.count += rhs.count
        self.max_occurences = max(self.max_occurences, rhs.max_occurences)


def update_substructure_metrics(
    prot: Protein,
    # First key is the InterPro element type, second key is the specific element ID
    metrics: Dict[str, Dict[str, SubstructureMetrics]],
):
    """Update the metrics dictionary with the counts from a single protein"""
    filtered_entries = [
        x
        for x in prot.entries
        if (x.element_type in have_representative and x.representative == "true")
        or (x.element_type not in have_representative)
    ]
    counts = defaultdict(int)
    for entry in filtered_entries:
        counts[entry.id] += 1

    for entry in filtered_entries:
        count = counts[entry.id]

        metrics[entry.element_type][entry.id].update_from_interpro(
            entry,
            int(prot.length),
            count,
        )
    # Now for secondary structures.
    counts = defaultdict(int)
    for ss in prot.secondary_structs:
        counts[ss.dssp_type.value] += 1

    for ss in prot.secondary_structs:
        count = counts[ss.dssp_type.value]
        metrics["secondary_struct"][DSSP_TO_NAME[ss.dssp_type.value]].update_from_secondary_struct(
            ss,
            int(prot.length),
            count,
        )


def calc_summaries(
    prots: Iterable[Protein],
    labels_path: str = "/weka/scratch/weka/kellislab/rcalef/data/interpro/102.0/label_sets/",
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, SubstructureMetrics]]]:
    # Define label sets for elements and initialize metric dicts
    substructure_metrics = defaultdict(dict)
    for interpro_type in want_types:
        labels = pd.read_table(os.path.join(labels_path, f"{interpro_type}.labels.tsv"))
        for _, row in labels.iterrows():
            substructure_metrics[interpro_type][row.interpro_id] = SubstructureMetrics(
                row.element_name,
                row.interpro_id,
            )
    for dssp_id, dssp_name in enumerate(DSSP_TO_NAME):
        substructure_metrics["secondary_struct"][dssp_name] = SubstructureMetrics(
            dssp_name,
            dssp_id,
        )

    prot_summaries = []
    for prot in prots:
        update_substructure_metrics(prot, substructure_metrics)
        prot_summaries.append(summarize_protein(prot))

    return pd.concat(prot_summaries), substructure_metrics


def merge_summaries(
    prot_summaries: List[pd.DataFrame],
    substructure_summaries: List[Dict[str, Dict[str, SubstructureMetrics]]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Merge the protein-centric summaries.
    merged_prot_summaries = (
        pd.concat(prot_summaries).rename_axis(index="uniprot_id").reset_index()
    )

    # Merge the substructure-centric summaries. More involved due to the nested dicts and
    # combining averages.
    merged_substructure_summaries = substructure_summaries[0]
    for summ in substructure_summaries[1:]:
        for element_type, type_metrics in summ.items():
            for element_id, metrics in type_metrics.items():
                merged_substructure_summaries[element_type][element_id].merge(metrics)

    # Convert SubstructureMetrics to dicts for easier dataframe construction
    for element_type, type_metrics in merged_substructure_summaries.items():
        for element_id, metrics in type_metrics.items():
            type_metrics[element_id] = metrics.__dict__

    # Convert nested dicts to dataframes.
    ret_dfs = {}
    for element_type, type_metrics in merged_substructure_summaries.items():
        ret_dfs[element_type] = (
            pd.DataFrame(type_metrics)
            .transpose()
            .reset_index(drop=True)
        )
    return merged_prot_summaries, ret_dfs
