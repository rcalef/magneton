import os

from dataclasses import dataclass, field, replace
from functools import partial
from typing import List, Set, Tuple

import torch

from esm.models.esmc import ESMC
from esm.sdk.api import (
    ESMProtein,
    ESMProteinTensor,
    LogitsConfig,
)
from esm.tokenization import get_esmc_model_tokenizers
from esm.utils.misc import stack_variable_length_tensors
from esm.utils.sampling import _BatchedESMProteinTensor
from tqdm import tqdm

from magneton.config import DataConfig, TrainingConfig
from magneton.data.meta_dataset import MetaDataset
from magneton.data.substructure import LabeledSubstructure
from magneton.embedders.base_embedder import BaseConfig, BaseDataModule, BaseEmbedder
from magneton.types import DataType
from magneton.utils import get_chunk_idxs


@dataclass
class SubstructureBatch:
    substructures: List[List[LabeledSubstructure]]

    def to(self, device: str):
        for i in range(len(self.substructures)):
            for j in range(len(self.substructures[i])):
                self.substructures[i][j] = self.substructures[i][j].to(device)
        return self

    def total_length(self) -> int:
        return sum(map(len, self.substructures))


@dataclass
class ESMCDataElem:
    tokenized_seq: torch.Tensor
    substructures: List[LabeledSubstructure]


@dataclass
class ESMCBatch(SubstructureBatch):
    tokenized_seq: _BatchedESMProteinTensor

    def to(self, device: str):
        super().to(device)
        self.tokenized_seq = self.tokenized_seq.to(device)
        return self


class ESMCDataSet(MetaDataset):
    def __init__(
        self,
        data_config: DataConfig,
    ):
        super().__init__(
            data_config=data_config,
            want_datatypes=[DataType.SEQ, DataType.SUBSTRUCT],
            load_fasta_in_mem=True,
        )
        self.tokenizer = get_esmc_model_tokenizers()

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx: int) -> ESMCDataElem:
        elem = self._prot_to_elem(self.dataset[idx])
        return ESMCDataElem(
            tokenized_seq=torch.tensor(self.tokenizer.encode(elem.seq)),
            substructures=elem.substructures,
        )


def esmc_collate(
    entries: List[ESMCDataElem],
    pad_id: int,
    drop_empty_substructures: bool = True,
) -> ESMCBatch:
    """
    Collate the entries into a batch.
    """
    if drop_empty_substructures:
        entries = [e for e in entries if len(e.substructures) > 0]
        if len(entries) == 0:
            entries = [ESMCDataElem(tokenized_seq=torch.zeros(128, dtype=torch.long), substructures=[])]

    # print(len(entries), [x.tokenized_seq for x in entries])

    padded_tensor = stack_variable_length_tensors(
        [x.tokenized_seq for x in entries],
        constant_value=pad_id,
    )
    substructs = [x.substructures for x in entries]
    return ESMCBatch(
        tokenized_seq=_BatchedESMProteinTensor(sequence=padded_tensor),
        substructures=substructs,
    )


class ESMCDataModule(BaseDataModule):
    def __init__(
        self,
        data_config: DataConfig,
        train_config: TrainingConfig,
    ):
        super().__init__(data_config, train_config)

    def _get_loader(self, dataset: ESMCDataSet) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            shuffle=True,
        )

    def _get_split_info(self, split: str) -> Tuple[str, str]:
        if split == "all":
            return self.data_config.data_dir, self.data_config.prefix
        else:
            return (
                os.path.join(
                    self.data_config.data_dir, f"{split}_sharded"
                ),
                f"swissprot.with_ss.{split}"
            )

    def _get_dataloader(
        self,
        split: str,
        **kwargs,
    ) -> torch.utils.data.DataLoader:
        data_dir, prefix = self._get_split_info(split)
        config = replace(
            self.data_config,
            data_dir=data_dir,
            prefix=prefix,
        )
        dataset = ESMCDataSet(
            data_config=config,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            collate_fn=partial(esmc_collate, pad_id=dataset.tokenizer.pad_token_id),
            num_workers=3,
            **kwargs,
        )

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


@dataclass
class ESMCConfig(BaseConfig):
    weights_path: str = field(kw_only=True)
    use_flash_attn: bool = field(kw_only=True, default=False)
    rep_layer: int = field(kw_only=True, default=35)
    max_seq_length: int = field(kw_only=True, default=2048)


class ESMCEmbedder(BaseEmbedder):
    def __init__(
        self,
        config: ESMCConfig,
        frozen: bool = True,
    ):
        super().__init__(config)

        self.model = ESMC(
            d_model=1152,
            n_heads=18,
            n_layers=36,
            tokenizer=get_esmc_model_tokenizers(),
            use_flash_attn=config.use_flash_attn,
        ).eval()

        if frozen:
            for param in self.parameters():
                param.requires_grad = False

        state_dict = torch.load(
            os.path.join(
                config.weights_path,
                "data",
                "weights",
                "esmc_600m_2024_12_v0.pth",
            ),
        )
        self.model.load_state_dict(state_dict)
        self.max_len = config.max_seq_length
        self.rep_layer = config.rep_layer
        self.device = config.device

    @torch.no_grad()
    def _get_embedding(
        self,
        protein_tensor: _BatchedESMProteinTensor,
    ) -> torch.Tensor:
        logits_config = LogitsConfig(
            sequence=True,
            return_embeddings=False,
            return_hidden_states=True,
        )
        logits_out = self.model.logits(protein_tensor, logits_config)
        return self.model.transformer.norm(logits_out.hidden_states)[self.rep_layer]

    @torch.no_grad()
    def embed_batch(self, batch: ESMCBatch) -> torch.Tensor:
        """Embed a batch of pre-tokenized protein sequences"""
        batch.tokenized_seq = batch.tokenized_seq.to(self.device)

        return self._get_embedding(batch.tokenized_seq)[:, 1:-1, :]

    # the following two functions are deprecated for the current data module setup
    @torch.no_grad()
    def embed_single_protein(self, seq: str) -> torch.Tensor:
        """Process a single protein sequence through ESM"""
        protein = ESMProtein(sequence=seq)
        protein_tensor = self.model.encode(protein)

        seq_len = len(seq) + 2  # +2 for EOS and BOS
        if seq_len <= self.max_len:
            ret = self._get_embedding(protein_tensor)
        else:
            # Get chunk indices for long sequences
            idxs = get_chunk_idxs(seq_len, max_len=self.config.max_seq_length)
            outputs = []
            for start, end in idxs:
                sub_tensor = ESMProteinTensor(
                    sequence=protein_tensor.sequence[start:end]
                ).to(self.device)
                outputs.append(self._get_embedding(sub_tensor))
            ret = torch.cat(outputs, dim=1)

        return ret.squeeze()[1 : len(seq) + 1]

    @torch.no_grad()
    def embed_sequences(self, sequences: List[str]) -> List[torch.Tensor]:
        """Embed multiple protein sequences"""
        all_embeddings = []

        for seq in tqdm(sequences, desc="Processing sequences"):
            try:
                embedding = self.embed_single_protein(seq)
                all_embeddings.append(embedding)
            except Exception as e:
                print(f"Error processing sequence: {str(e)}")
                continue

        return all_embeddings

    def get_embed_dim(self):
        return 1152

    @classmethod
    def get_required_input_type(cls) -> Set[DataType]:
        return {DataType.SEQ}

    @classmethod
    def model_name(cls) -> str:
        return "ESM-C"
