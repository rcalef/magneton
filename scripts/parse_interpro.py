import fire

from magneton.io.interpro import parse_from_xml
from magneton.io.internal import shard_proteins


def parse_and_write(
    input_path: str,
    output_dir: str,
    print_iter: int = 10000,
    prots_per_file: int = 500000,
):
    iter = parse_from_xml(input_path, print_iter)
    shard_proteins(iter, output_dir, prots_per_file=prots_per_file)


if __name__ == "__main__":
    fire.Fire(parse_and_write)
