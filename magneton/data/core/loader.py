import os
from dataclasses import replace
from functools import partial

import torch

from torchdata.nodes import (
    BaseNode,
    MapStyleWrapper,
    Filter,
)
from torch.utils.data import (
    DistributedSampler,
    RandomSampler,
    SequentialSampler,
)

from magneton.config import DataConfig
from magneton.types import DataType

from .core_dataset import (
    CoreDataset,
    DataElement,
)

def max_len_filter(
    x: DataElement,
    max_len: int,
) -> bool:
    """Keep proteins below max_len."""
    return x.length <= max_len

def get_core_node(
    data_config: DataConfig,
    want_datatypes: list[DataType],
    split: str = "train",
    max_len: int | None = 2048,
    load_fasta_in_mem: bool = True,
    seed: int = 42,
    shuffle: bool = False,
    distributed: bool = False,
    drop_last: bool = False,
) -> BaseNode:
    if split != "all":
        split_dir = os.path.join(data_config.data_dir, f"{split}_sharded")
        prefix = f"swissprot.with_ss.{split}"
        data_config = replace(
            data_config,
            data_dir=split_dir,
            prefix=prefix,
        )

    prot_dataset = CoreDataset(
        data_config=data_config,
        want_datatypes=want_datatypes,
        load_fasta_in_mem=load_fasta_in_mem,
    )
    if distributed:
        sampler = DistributedSampler(
            dataset=prot_dataset,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
    else:
        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(seed)
            sampler = RandomSampler(
                data_source=prot_dataset,
                replacement=False,
                generator=generator,
            )
        else:
            sampler = SequentialSampler(
                data_source=prot_dataset,
            )
    node = MapStyleWrapper(map_dataset=prot_dataset, sampler=sampler)
    if max_len is not None:
        node = Filter(node, filter_fn=partial(max_len_filter, max_len=max_len))

    return node
