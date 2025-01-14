import os

from collections import defaultdict
from typing import (
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
)

import pandas as pd

from magneton.types import (
    Protein,
    DSSP_TO_NAME,
)

from magneton.types import DSSP_TO_NAME

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
        "ss_cov": ss_cov,
    }
    ret.update(interpro_counts.to_dict())
    ret.update(ss_counts.to_dict())

    return pd.DataFrame(ret, index=[prot.uniprot_id])


def update_substructure_metrics(
    prot: Protein,
    # First key is the InterPro element type, second key is the specific element name
    metrics: Dict[str, Dict[str, float | int]],
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

    seen = set()
    for entry in filtered_entries:
        entry_dict = metrics[entry.element_type][entry.element_name]
        entry_len = sum([end - start for start, end in entry.positions])

        if entry_dict["min_len"] is None or entry_len < entry_dict["min_len"]:
            entry_dict["min_len"] = entry_len
        if entry_dict["max_len"] is None or entry_len > entry_dict["max_len"]:
            entry_dict["max_len"] = entry_len

        # Update running average of length.
        entry_dict["avg_len"] = increment_average(
            entry_dict["avg_len"],
            entry_len,
            entry_dict["count_total"],
        )

        # Update running average of coverage.
        entry_dict["avg_coverage"] = increment_average(
            entry_dict["avg_coverage"],
            entry_len / int(prot.length),
            entry_dict["count_total"],
        )

        entry_dict["count_total"] += 1

        if entry.id not in seen:
            seen.add(entry.id)

            entry_dict["count_unique"] += 1
            this_count = counts[entry.id]
            if this_count > entry_dict["max_count"]:
                entry_dict["max_count"] = this_count


def calc_summaries(
    prots: Iterable[Protein],
    labels_path: str = "/weka/scratch/weka/kellislab/rcalef/data/interpro/102.0/label_sets/",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Define label sets for elements and initialize metric dicts
    substructure_metrics = defaultdict(dict)
    for interpro_type in want_types:
        labels = pd.read_table(os.path.join(labels_path, f"{interpro_type}.labels.tsv"))
        for _, row in labels.iterrows():
            substructure_metrics[interpro_type][row.element_name] = {
                "interpro_id": row.interpro_id,
                "count_total": 0,
                "count_unique": 0,
                "max_count": 0,
                "min_len": None,
                "max_len": None,
                "avg_len": None,
                "avg_coverage": None,
            }
    prot_summaries = []
    for prot in prots:
        update_substructure_metrics(prot, substructure_metrics)
        prot_summaries.append(summarize_protein(prot))

    return pd.concat(prot_summaries), substructure_metrics


def merge_summary_dicts(
    lhs: Dict[str, float | int],
    rhs: Dict[str, float | int],
):
    if rhs["min_len"] is not None:
        if lhs["min_len"] is None or rhs["min_len"] < lhs["min_len"]:
            lhs["min_len"] = rhs["min_len"]

    if rhs["max_len"] is not None:
        if lhs["max_len"] is None or rhs["max_len"] > lhs["max_len"]:
            lhs["max_len"] = rhs["max_len"]

    new_avg_len, _ = combine_averages(
        lhs["avg_len"], lhs["count_total"], rhs["avg_len"], rhs["count_total"]
    )
    lhs["avg_len"] = new_avg_len
    new_avg_cov, new_count = combine_averages(
        lhs["avg_coverage"], lhs["count_total"], rhs["avg_coverage"], rhs["count_total"]
    )
    lhs["avg_coverage"] = new_avg_cov

    lhs["count_total"] = new_count
    lhs["count_unique"] += rhs["count_unique"]
    lhs["max_count"] = max(lhs["max_count"], rhs["max_count"])

    return lhs


def merge_summaries(
    prot_summaries: List[pd.DataFrame],
    substructure_summaries: List[Dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    # Merge the protein-centric summaries.
    merged_prot_summaries = (
        pd.concat(prot_summaries).rename_axis(index="uniprot_id").reset_index()
    )

    # Merge the substructure-centric summaries. More involved due to the nested dicts and
    # combining averages.
    merged_substructure_summaries = substructure_summaries[0]
    for summ in substructure_summaries[1:]:
        for element_type, type_metrics in summ.items():
            for element_name, metrics in type_metrics.items():
                merged_substructure_summaries[element_type][element_name] = (
                    merge_summary_dicts(
                        merged_substructure_summaries[element_type][element_name],
                        metrics,
                    )
                )
    # Convert nested dicts to dataframes.
    ret_dfs = {}
    for interpro_type, metrics in merged_substructure_summaries.items():
        ret_dfs[interpro_type] = (
            pd.DataFrame(metrics)
            .transpose()
            .rename_axis(index="element_name")
            .reset_index()
        )
    return merged_prot_summaries, ret_dfs
