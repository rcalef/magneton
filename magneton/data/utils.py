import os
from typing import Set

from torch.utils.data import DataLoader

from magneton.config import DataConfig
from magneton.types import DataType
from magneton.data.meta_dataset import (
    MetaDataset,
    collate_meta_datasets,
    collate_sequence_datasets,
    SequenceOnlyDataset,
)
def get_dataloader(
    config: DataConfig,
    data_types: Set[DataType],
    batch_size: int,
    num_workers: int = 0,
) ->DataLoader:
    """
    Get a dataloader for the specified data types.
    """
    dataset = MetaDataset(
        input_path=config.data_dir,
        want_datatypes=data_types,
        fasta_path=config.fasta_path,
        compression=config.compression,
        prefix=config.prefix,
        labels_path=config.labels_path,
        want_interpro_types=config.interpro_types,
        load_fasta_in_mem=True,
    )
    return DataLoader(
        dataset=MetaDataset,
        batch_size=batch_size,
        collate_fn=collate_meta_datasets,
        num_workers=num_workers,
    )

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
        num_workers=num_workers,
    )