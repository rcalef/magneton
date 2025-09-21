from functools import partial
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import (
    Dataset,
    DistributedSampler,
    RandomSampler,
    SequentialSampler,
    Subset,
)
from torchdata.nodes import (
    BaseNode,
    Batcher,
    Loader,
    Mapper,
    Prefetcher,
    SamplerWrapper,
)

from magneton.config import DataConfig
from magneton.types import DataType

from .core import (
    get_core_dataset,
)
from .evals import (
    PEER_TASK_TO_CONFIGS,
    TASK_GRANULARITY,
    TASK_TO_TYPE,
    BioLIP2Module,
    ContactPredictionModule,
    DeepFriModule,
    DeepLocModule,
    FlipModule,
    PeerDataModule,
    ThermostabilityModule,
)
from .model_specific import (
    ESM2TransformNode,
    ESMCTransformNode,
    ProSSTTransformNode,
    SaProtTransformNode,
)

model_data = {
    "esm2": (ESM2TransformNode, [DataType.SEQ]),
    "esmc": (ESMCTransformNode, [DataType.SEQ]),
    "prosst": (ProSSTTransformNode, [DataType.SEQ, DataType.STRUCT]),
    "saprot": (SaProtTransformNode, [DataType.SEQ, DataType.STRUCT]),
}


def filter_and_sample(
    dataset: Dataset,
    shuffle: bool,
    max_len: int | None = 2048,
    distributed: bool = False,
    seed: int = 42,
    drop_last: bool = True,
) -> BaseNode:
    if max_len is not None:
        valid_indices = []
        for i in range(len(dataset)):
            if dataset[i].length < max_len:
                valid_indices.append(i)
        dataset = Subset(dataset, valid_indices)
        print(f"remaining samples after length filter: {len(valid_indices)}")

    if distributed:
        sampler = DistributedSampler(
            dataset=dataset,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
    else:
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
    return Mapper(sampler_node, dataset.__getitem__)


class MagnetonDataModule(L.LightningDataModule):
    """DataModule for substructure-aware fine-tuning."""

    def __init__(
        self,
        data_config: DataConfig,
        model_type: str,
        distributed: bool = False,
        max_len: int | None = 2048,
    ):
        super().__init__()
        self.data_config = data_config
        transform_cls, want_datatypes = model_data[model_type]
        self.transform_cls = transform_cls
        self.want_datatypes = want_datatypes + [DataType.SUBSTRUCT]
        self.distributed = distributed
        self.max_len = max_len

    def _get_dataloader(
        self,
        split: str,
        shuffle: bool,
    ) -> Loader:
        dataset = get_core_dataset(
            data_config=self.data_config,
            want_datatypes=self.want_datatypes,
            split=split,
        )
        node = filter_and_sample(
            dataset=dataset,
            distributed=self.distributed,
            max_len=self.max_len,
            shuffle=shuffle,
        )
        node = self.transform_cls(
            node,
            self.data_config.data_dir,
            **self.data_config.model_specific_params,
        )
        collate_fn = node.get_collate_fn()

        node = Batcher(node, batch_size=self.data_config.batch_size)
        node = Prefetcher(node, prefetch_factor=16)
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
        num_workers: int = 32,
    ):
        super().__init__()
        transform_cls, _ = model_data[model_type]
        self.transform_cls = transform_cls
        self.task = task
        self.task_type = TASK_TO_TYPE[task]
        self.data_config = data_config
        self.data_dir = Path(data_dir)
        self.distributed = distributed
        self.max_len = max_len
        self.num_workers = num_workers

        if task in ["GO:BP", "GO:CC", "GO:MF", "EC"]:
            task = task.replace("GO:", "")
            self.module = DeepFriModule(
                task,
                self.data_dir,
                struct_template=self.data_config.struct_template,
                num_workers=num_workers,
            )
            self.task_granularity = TASK_GRANULARITY.PROTEIN_CLASSIFICATION
        elif task in PEER_TASK_TO_CONFIGS:
            self.module = PeerDataModule(
                task,
                self.data_dir,
            )
            self.task_granularity = TASK_GRANULARITY.PROTEIN_CLASSIFICATION
        elif task.startswith("saprot"):
            if task == "saprot_thermostability":
                self.module = ThermostabilityModule(
                    self.data_dir / "saprot_processed" / "Thermostability",
                    struct_template=self.data_config.struct_template,
                    num_workers=num_workers,
                )
                self.task_granularity = TASK_GRANULARITY.PROTEIN_CLASSIFICATION
            elif task in ["saprot_binloc", "saprot_subloc"]:
                num_labels = 2 if task == "saprot_binloc" else 10
                self.module = DeepLocModule(
                    self.data_dir / "saprot_processed" / "DeepLoc",
                    struct_template=self.data_config.struct_template,
                    num_labels=num_labels,
                    num_workers=num_workers,
                )
                self.task_granularity = TASK_GRANULARITY.PROTEIN_CLASSIFICATION

        elif task == "FLIP_bind":
            self.module = FlipModule(
                data_dir=self.data_dir / "FLIP_bind",
                struct_template=self.data_config.struct_template,
                num_workers=num_workers,
            )
            self.task_granularity = TASK_GRANULARITY.RESIDUE_CLASSIFICATION
        elif task == "contact_prediction":
            self.module = ContactPredictionModule(
                data_dir=self.data_dir / "saprot_processed" / "Contact",
                num_workers=num_workers,
            )
            self.task_granularity = TASK_GRANULARITY.CONTACT_PREDICTION
        elif task in ["biolip_binding", "biolip_catalytic"]:
            self.module = BioLIP2Module(
                data_dir=self.data_dir / "struct_token_bench",
                task=task.replace("biolip_", ""),
                num_workers=num_workers,
            )
            self.task_granularity = TASK_GRANULARITY.RESIDUE_CLASSIFICATION

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
        node = filter_and_sample(
            dataset=dataset,
            distributed=self.distributed,
            max_len=self.max_len,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
        )

        this_data_dir = self.data_dir / self.task / split
        if not this_data_dir.exists():
            this_data_dir.mkdir(parents=True, exist_ok=True)
        # Allocate half the workers to model-specific transforms, since
        # this is typically where most of the work is.
        node = self.transform_cls(
            node,
            this_data_dir,
            num_workers=self.num_workers // 2,
            **self.data_config.model_specific_params,
        )
        labels_mode = None
        if self.task_granularity == TASK_GRANULARITY.PROTEIN_CLASSIFICATION:
            labels_mode = "stack"
        elif self.task_granularity == TASK_GRANULARITY.RESIDUE_CLASSIFICATION:
            labels_mode = "cat"
        elif self.task_granularity == TASK_GRANULARITY.CONTACT_PREDICTION:
            labels_mode = "pad"
        else:
            raise ValueError(f"unexpected task granularity: {self.task_granularity}")

        collate_fn = node.get_collate_fn(
            labels_mode=labels_mode,
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
