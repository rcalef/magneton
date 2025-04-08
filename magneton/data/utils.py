from typing import Set

from torch.utils.data import DataLoader

from magneton.config import DataConfig
from magneton.types import DataType
from magneton.data.meta_dataset import (
    MetaDataset,
    collate_meta_datasets,
)


def get_dataloader(
    config: DataConfig,
    data_types: Set[DataType],
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
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
        struct_template=config.struct_template,
        load_fasta_in_mem=True,
    )
    return DataLoader(
        dataset=MetaDataset,
        batch_size=batch_size,
        collate_fn=collate_meta_datasets,
        num_workers=num_workers,
    )
