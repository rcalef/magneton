import pickle

import fire

from magneton.interpro_parsing import parse_from_xml

def parse_and_write(
    input_path: str,
    output_path: str,
    print_iter: int = 10000,
):
    with open(output_path, "wb") as out_fh:
        for prot in parse_from_xml(input_path, print_iter):
            pickle.dump(prot, out_fh)

if __name__ == "__main__":
    fire.Fire(parse_and_write)
