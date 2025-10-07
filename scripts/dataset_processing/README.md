This directory contains scripts used for creating the Magneton datasets. These scripts were originally run in this order:
  1. `parse_interpro.py` - used to parse the original InterPro XML file into sharded JSONL files.
  2. `filter_proteins.py` - used in combination with the SwissProt FASTA file to filter the full InterPro dataset generated above (which contains all of UniProt) to the SwissProt subset.
  3. `add_secondary_structs.py` - used along with AFDB's SwissProt release to add secondary structure annotations to the sharded SwissProt dataset.