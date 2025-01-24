import os

import fire
import numpy as np
import pandas as pd


def make_interpro_labels(
    interpro_path: str,
):
    labels = pd.read_table(os.path.join(interpro_path, "entry.list.tsv")).rename(
        columns={
            "ENTRY_AC": "id",
            "ENTRY_TYPE": "element_type",
            "ENTRY_NAME": "element_name",
        }
    )
    labels.head()

    # Start with all
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

    for element_type in want_types:
        elem_labels = labels.query("element_type == @element_type")
        assert not elem_labels.element_name.duplicated().all()
        label_set = elem_labels.sort_values("id")
        label_df = pd.DataFrame(
            {
                "label": np.arange(len(label_set)),
                "element_name": label_set.element_name,
                "interpro_id": label_set.id,
            }
        )
        with open(
            os.path.expanduser(
                os.path.join(interpro_path, "label_sets", f"{element_type}.labels.tsv")
            ),
            "w",
        ) as fh:
            label_df.to_csv(fh, sep="\t", index=False)


if __name__ == "__main__":
    fire.Fire(make_interpro_labels)
