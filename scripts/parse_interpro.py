import bz2
import os
import pickle

import fire

from magneton.interpro_parsing import parse_from_xml

def parse_and_write(
    input_path: str,
    output_dir: str,
    print_iter: int = 10000,
    prots_per_file: int = 500000
):
    prev_id = "0"
    index_path = os.path.join(output_dir, "index.txt")
    with open(index_path, "w") as index_fh:
        curr_file_prots = 0
        curr_file_num = 0
        curr_fh = bz2.open(os.path.join(output_dir, f"parsed_proteins.{curr_file_num}.pkl.bz2"), "wb")
        for prot in parse_from_xml(input_path, print_iter):
            assert prev_id < prot.uniprot_id, f"{prev_id} !< {prot.uniprot_id}"
            prev_id = prot.uniprot_id

            if curr_file_prots == 0:
                index_fh.write(f"{curr_file_num}\t{prot.uniprot_id}\n")

            curr_file_prots += 1
            pickle.dump(prot, curr_fh)
            if curr_file_prots == prots_per_file:
                curr_fh.close()
                print(f"completed file {curr_file_num}, starting file {curr_file_num+1}")

                curr_file_num += 1
                curr_file_prots = 0
                curr_fh = bz2.open(os.path.join(output_dir, f"parsed_proteins.{curr_file_num}.pkl.bz2"), "wb")
    curr_fh.close()

if __name__ == "__main__":
    fire.Fire(parse_and_write)
