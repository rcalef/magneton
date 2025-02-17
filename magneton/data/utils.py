import os
from typing import Set

from torch.utils.data import DataLoader

from magneton.config import DataConfig
from magneton.constants import DataType
from magneton.data.sequence_only import (
    collate_sequence_datasets,
    SequenceOnlyDataset,
)
def get_dataloader(
    config: DataConfig,
    data_types: Set[DataType],
    batch_size: int,
) ->DataLoader:
    """
    Get a dataloader for the specified data types.
    """
    # TODO: Add support for other data types, composition
    if DataType.SEQ in data_types:
        dataset = SequenceOnlyDataset(
            input_path=config.data_dir,
            fasta_path=config.fasta_path,
            compression=config.compression,
            prefix=config.prefix,
        )
        collate_fn = collate_sequence_datasets
    else:
        raise ValueError(f"Unsupported data type: {data_types}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        #num_workers=max(os.cpu_count() - 1, 1),
        num_workers=0,
    )