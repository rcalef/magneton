from typing import List

import pandas as pd
from pdbecif.mmcif_io import CifFileReader

from magneton.custom_types import (
    DsspType,
    SecondaryStructure,
    MMCIF_TO_DSSP,
)

def mmcif_to_secondary_structs(
    path: str
) -> List[SecondaryStructure]:
    """Given path to mmCIF file, return list of secondary structure annotations"""
    want_cols = ["beg_label_seq_id", "end_label_seq_id", "conf_type_id"]
    cfr = CifFileReader()
    cif_obj = cfr.read(path)

    keys = list(cif_obj.keys())
    assert len(keys) == 1
    cif_obj = cif_obj[keys[0]]

    # Some mmCIF files don't have secondary structure annotations, e.g. very
    # short peptides (e.g. A0A0C5B5G6)
    if "_struct_conf" not in cif_obj:
        return []

    # Some mmCIF files have a single secondary structure annotation, which requires
    # special handling.
    num_ss = len(cif_obj["_struct_conf"]["beg_auth_asym_id"])
    if num_ss == 1:
        ss_df = pd.DataFrame(cif_obj["_struct_conf"], index=[0])
    else:
        ss_df = pd.DataFrame(cif_obj["_struct_conf"])

    structs = []
    for begin, end, dssp_type in ss_df[want_cols].itertuples(index=False):
        type_id = MMCIF_TO_DSSP[dssp_type]
        structs.append(SecondaryStructure(
            start=int(begin),
            # Convert end to exclusive coordinates to better mesh with Python
            end=int(end)+1,
            dssp_type=DsspType(type_id),
        ))
    return structs