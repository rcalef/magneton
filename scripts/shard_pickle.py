import fire

from magneton.io.internal import (
    parse_from_pkl,
    shard_proteins,
)

def shard_pkl(
    input_path: str,
    output_dir: str,
    prefix: str = "sharded_proteins",
    prots_per_file: int = 500000,
):
    iter = parse_from_pkl(input_path, compression="bz2")
    shard_proteins(iter, output_dir, prefix=prefix, prots_per_file=prots_per_file)


if __name__ == "__main__":
    fire.Fire(shard_pkl)
