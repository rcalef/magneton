from enum import StrEnum
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import (
    DistributedSampler,
    RandomSampler,
    SequentialSampler,
)
from torchdata.nodes import (
    Batcher,
    Filter,
    Header,
    Loader,
    Mapper,
    SamplerWrapper,
)

from magneton.config import DataConfig
from magneton.types import DataType

from .core import (
    get_core_node,
    DeepFriModule,
    FlipModule,
    PeerDataModule,
    PEER_TASK_TO_CONFIGS,
    #WorkshopDataModule,
    #TASK_TO_CONFIGS,
)
from .model_specific import (
    ESMCTransformNode,
    # ProSSTTransformNode,
)

model_data = {
    "esmc": (ESMCTransformNode, [DataType.SEQ]),
    # "prosst": (ProSSTTransformNode, [DataType.SEQ, DataType.STRUCT])
}

class TASK_TYPE(StrEnum):
    PROTEIN_CLASSIFICATION = "protein_classification"
    RESIDUE_CLASSIFICATION = "residue_classification"


class MagnetonDataModule(L.LightningDataModule):
    """DataModule for substructure-aware fine-tuning."""
    def __init__(
        self,
        data_config: DataConfig,
        model_type: str,
        distributed: bool = False,
    ):
        super().__init__()
        self.data_config = data_config
        transform_cls, want_datatypes = model_data[model_type]
        self.transform_cls = transform_cls
        self.want_datatypes = want_datatypes + [DataType.SUBSTRUCT]
        self.distributed = distributed

    def _get_dataloader(
        self,
        split: str,
        **kwargs,
    ) -> Loader:
        node = get_core_node(
            data_config=self.data_config,
            want_datatypes=self.want_datatypes,
            split=split,
            distributed=self.distributed,
            **kwargs,
        )
        node = self.transform_cls(
            node,
            self.data_config.data_dir,
            **self.data_config.model_specific_params,
        )
        collate_fn = node.get_collate_fn()

        node = Batcher(node, batch_size=self.data_config.batch_size)
        node = Mapper(node, collate_fn)

        return Loader(node)

    def train_dataloader(self):
        return self._get_dataloader(
            "train",
            shuffle=True,
        )

    def val_dataloader(self):
        return self._get_dataloader(
            "val",
            shuffle=False,
        )

    def test_dataloader(self):
        return self._get_dataloader(
            "test",
            shuffle=False,
        )

    def predict_dataloader(self):
        return self._get_dataloader(
            "all",
            shuffle=False,
        )

class SupervisedDownstreamTaskDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_config: DataConfig,
        task: str,
        data_dir: str,
        model_type: str,
        max_len: int | None = 2048,
        distributed: bool = False,
        num_workers: int = 32
    ):
        super().__init__()
        transform_cls, _ = model_data[model_type]
        self.transform_cls = transform_cls
        self.task = task
        self.data_config = data_config
        self.data_dir = Path(data_dir)
        self.distributed = distributed
        self.max_len = max_len

        if task in ["GO:BP", "GO:CC", "GO:MF", "EC"]:
            task = task.replace("GO:", "")
            self.module = DeepFriModule(
                task,
                self.data_dir,
                struct_template=self.data_config.struct_template,
                num_workers=num_workers,
            )
            self.task_type = TASK_TYPE.PROTEIN_CLASSIFICATION
        elif task in PEER_TASK_TO_CONFIGS:
            self.module = PeerDataModule(
                task,
                self.data_dir,
            )
            self.task_type = TASK_TYPE.PROTEIN_CLASSIFICATION
        # elif task in TASK_TO_CONFIGS:
        #     self.module = WorkshopDataModule(
        #         task,
        #         self.data_dir,
        #     )
        elif task == "FLIP_bind":
            self.module = FlipModule(
                data_dir=self.data_dir / "FLIP_bind",
                struct_template=self.data_config.struct_template,
                num_workers=num_workers,
            )
            self.task_type = TASK_TYPE.RESIDUE_CLASSIFICATION

        else:
            raise ValueError(f"unknown eval task: {task}")

    def num_classes(self) -> int:
        return self.module.num_classes()

    def _get_dataloader(
        self,
        split: str,
        shuffle: bool = False,
        seed: int = 42,
        drop_last: bool = True,
    ) -> Loader:
        dataset = self.module.get_dataset(split)
        if self.distributed:
            print("using distributed sampler")
            sampler = DistributedSampler(
                dataset=dataset,
                shuffle=shuffle,
                seed=seed,
                drop_last=drop_last,
            )
        else:
            print("not using distributed sampler")
            if shuffle:
                generator = torch.Generator()
                generator.manual_seed(seed)
                sampler = RandomSampler(
                    data_source=dataset,
                    replacement=False,
                    generator=generator,
                )
            else:
                sampler = SequentialSampler(
                    data_source=dataset,
                )
        sampler_node = SamplerWrapper(
            sampler=sampler,
        )
        node = Mapper(sampler_node, dataset.__getitem__)
        if self.max_len is not None:
            node = Filter(node, filter_fn=lambda x: x.length < self.max_len)

        this_data_dir = self.data_dir / self.task / split
        node = self.transform_cls(
            node,
            this_data_dir,
            **self.data_config.model_specific_params,
        )
        collate_fn = node.get_collate_fn(
            stack_labels=self.task_type == TASK_TYPE.PROTEIN_CLASSIFICATION,
        )

        node = Batcher(node, batch_size=self.data_config.batch_size)
        node = Mapper(node, collate_fn)

        return Loader(node)

    def train_dataloader(self):
        return self._get_dataloader(
            "train",
            shuffle=True,
        )

    def val_dataloader(self):
        return self._get_dataloader(
            "val",
            shuffle=False,
        )

    def test_dataloader(self):
        return self._get_dataloader(
            "test",
            shuffle=False,
        )

    def predict_dataloader(self):
        return self._get_dataloader(
            "all",
            shuffle=False,
        )