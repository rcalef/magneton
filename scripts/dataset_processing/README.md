This directory contains scripts used for creating the Magneton datasets. These scripts were originally run in this order:
  1. `parse_interpro.py` - used to parse the original InterPro XML file into sharded JSONL files.
  2. `make_interpro_labels.py` - used to parse the original InterPro `entry.list.tsv` file into sharded a separate file of unique types per InterPro category.
  3. `curate_swissprot_set.ipynb` - used to curate the SwissProt subset of UniProt IDs used to filter the fill InterPro dataset.
  4. `filter_proteins.py` - used in combination with the file above to filter the full InterPro dataset generated above (which contains all of UniProt) to the SwissProt subset.
  5. `add_secondary_structs.py` - used along with AFDB's SwissProt release to add secondary structure annotations to the sharded SwissProt dataset.
  6. `calc_stats.py` - used to compute summary statistics of occurences of different substructure types.
  7. `create_label_subsets.ipynb` - used to create the subset of frequently occurring substructure types used as the label set for subsequent substructure-tuning experiments.