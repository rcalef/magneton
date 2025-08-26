import os

from dataclasses import dataclass, field, replace
from functools import partial
from typing import List, Literal, Set, Tuple

import torch

from esm.models.esmc import ESMC
from esm.sdk.api import (
    ESMProtein,
    ESMProteinTensor,
    LogitsConfig,
)
from esm.tokenization import get_esmc_model_tokenizers
from esm.utils.misc import stack_variable_length_tensors
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from magneton.config import DataConfig, TrainingConfig
from magneton.data.meta_dataset import MetaDataset
from magneton.data.substructure import LabeledSubstructure, SubstructureBatch
from magneton.embedders.base_embedder import BaseConfig, BaseDataModule, BaseEmbedder
from magneton.types import DataType
from magneton.utils import get_chunk_idxs


@dataclass
class ESMCDataElem:
    tokenized_seq: torch.Tensor
    substructures: List[LabeledSubstructure]
    prot_id: str


@dataclass
class ESMCBatch(SubstructureBatch):
    tokenized_seq: torch.Tensor

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
        # elem: BatchElem
        elem = self._prot_to_elem(self.dataset[idx])
        prot_id = elem.protein_id
        return ESMCDataElem(
            tokenized_seq=torch.tensor(self.tokenizer.encode(elem.seq)),
            substructures=elem.substructures,
            prot_id=prot_id,
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
            entries = [
                ESMCDataElem(
                    tokenized_seq=torch.zeros(128, dtype=torch.long),
                    substructures=[],
                    prot_id="",
                )
            ]

    # print(len(entries), [x.tokenized_seq for x in entries])

    padded_tensor = stack_variable_length_tensors(
        [x.tokenized_seq for x in entries],
        constant_value=pad_id,
    )
    substructs = [x.substructures for x in entries]
    prot_ids = [x.prot_id for x in entries]
    return ESMCBatch(
        tokenized_seq=padded_tensor,
        substructures=substructs,
        prot_ids=prot_ids,
    )


class ESMCDataModule(BaseDataModule):
    def __init__(
        self,
        data_config: DataConfig,
        train_config: TrainingConfig,
    ):
        super().__init__(data_config, train_config)

    def _get_split_info(self, split: str) -> Tuple[str, str]:
        if split == "all":
            return self.data_config.data_dir, self.data_config.prefix
        else:
            return (
                os.path.join(self.data_config.data_dir, f"{split}_sharded"),
                f"swissprot.with_ss.{split}",
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

ESMC_300M = "300m"
ESMC_600M = "600m"


@dataclass
class ESMCConfig(BaseConfig):
    model_size: Literal[ESMC_300M, ESMC_600M] = field(kw_only=True, default=ESMC_600M)
    weights_path: str = field(kw_only=True)
    use_flash_attn: bool = field(kw_only=True, default=False)
    rep_layer: int = field(kw_only=True, default=35)
    max_seq_length: int = field(kw_only=True, default=2048)
    mask_prob: float = field(kw_only=True, default=0.15)


class ESMCEmbedder(BaseEmbedder):
    def __init__(
        self,
        config: ESMCConfig,
        frozen: bool = True,
    ):
        super().__init__(config)

        self.max_len = config.max_seq_length
        self.rep_layer = config.rep_layer
        self.model_size = config.model_size

        if config.model_size == ESMC_600M:
            self.model = ESMC(
                d_model=1152,
                n_heads=18,
                n_layers=36,
                tokenizer=get_esmc_model_tokenizers(),
                use_flash_attn=config.use_flash_attn,
            )
            self.embed_dim=1152
        elif config.model_size == ESMC_300M:
            self.model = ESMC(
                d_model=960,
                n_heads=15,
                n_layers=30,
                tokenizer=get_esmc_model_tokenizers(),
                use_flash_attn=config.use_flash_attn,
            )
            self.embed_dim=960
        else:
            raise ValueError(f"unknown ESM-C model size: {config.model_size}")

        state_dict = torch.load(
            os.path.join(
                config.weights_path,
                "data",
                "weights",
                f"esmc_{config.model_size}_2024_12_v0.pth",
            ),
        )
        self.model.load_state_dict(state_dict)

        if frozen:
            self.model = self.model.eval()
            self._freeze()
        # Freeze everything after the layer we're
        # using for extracting representations to
        # avoid DDP errors.
        else:
            self._unfreeze(unfreeze_all=False)

        # For masking when calculating original MLM loss
        self.rng = torch.Generator().manual_seed(42)
        self.mask_prob = config.mask_prob

    def _freeze(self):
        for params in self.model.parameters():
            params.requires_grad = False

    def _unfreeze(
        self,
        unfreeze_all: bool = False,
    ):
        for param in self.model.parameters():
            param.requires_grad = True
        if not unfreeze_all:
            # Freeze layers that do not contribute to the embeddings
            # that we're extracting to prevent DDP exceptions.
            num_blocks = len(self.model.transformer.blocks)
            if self.rep_layer != num_blocks-1:
                for block in self.model.transformer.blocks[self.rep_layer+1:]:
                    for param in block.parameters():
                        param.requires_grad = False
            for param in self.model.sequence_head.parameters():
                param.requires_grad = False


    def _get_embedding(
        self,
        protein_tensor: torch.Tensor,
    ) -> torch.Tensor:
        logits_out = self.model(protein_tensor)

        return self.model.transformer.norm(logits_out.hidden_states[self.rep_layer])

    def embed_batch(self, batch: ESMCBatch) -> torch.Tensor:
        """Embed a batch of pre-tokenized protein sequences"""
        # Remove CLS token
        return self._get_embedding(batch.tokenized_seq)[:, 1:, :]

    # the following two functions are deprecated for the current data module setup
    @torch.inference_mode()
    def embed_single_protein(self, seq: str) -> torch.Tensor:
        """Process a single protein sequence through ESM"""
        protein = ESMProtein(sequence=seq)
        protein_tensor = self.model.encode(protein).sequence.unsqueeze(0)

        seq_len = len(seq) + 2  # +2 for EOS and BOS
        if seq_len <= self.max_len:
            ret = self._get_embedding(protein_tensor)
        else:
            # Get chunk indices for long sequences
            idxs = get_chunk_idxs(seq_len, max_len=self.config.max_seq_length)
            outputs = []
            for start, end in idxs:
                sub_tensor = protein_tensor[:, start:end]
                outputs.append(self._get_embedding(sub_tensor))
            ret = torch.cat(outputs, dim=1)

        return ret.squeeze()[1 : len(seq) + 1]

    @torch.inference_mode()
    def embed_sequences(self, sequences: List[str]) -> List[torch.Tensor]:
        """Embed multiple protein sequences"""
        all_embeddings = []

        for seq in tqdm(sequences, desc="Processing sequences"):
            try:
                embedding = self.embed_single_protein(seq)
                all_embeddings.append(embedding)
            except Exception as e:
                raise e
                print(f"Error processing sequence: {str(e)}")
                continue

        return all_embeddings

    def calc_original_loss(
        self,
        batch: ESMCBatch,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """NOTE: this modifies the original tokenized seq tensor, call last."""
        seqs = batch.tokenized_seq

        mask = get_seq_mask(
            seqs,
            pad_token_id=self.model.tokenizer.pad_token_id,
            bos_token_id=self.model.tokenizer.bos_token_id,
            eos_token_id=self.model.tokenizer.eos_token_id,
            rng=self.rng,
            mask_prob=self.mask_prob,
        )
        masked_idxs = torch.where(mask.reshape(-1))[0]

        # Store original mask values for loss calculation
        orig_values_flat = seqs.reshape(-1)[masked_idxs]

        seqs = seqs.masked_fill(mask, self.model.tokenizer.mask_token_id)

        logits = self.model.forward(seqs).sequence_logits

        # Flatten logits along batch dim and get logits for just masked positions
        masked_pos_logits = logits.reshape(-1, logits.shape[-1])[masked_idxs]
        return CrossEntropyLoss(reduction=reduction)(masked_pos_logits, orig_values_flat)

    def get_embed_dim(self):
        return self.embed_dim

    def model_name(self) -> str:
        return f"ESM-C_{self.model_size}"

    @classmethod
    def get_required_input_type(cls) -> Set[DataType]:
        return {DataType.SEQ}



def get_seq_mask(
    tokenized_seqs: torch.Tensor,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    rng: torch.Generator,
    mask_prob: float = 0.15,
) -> torch.Tensor:
    probs = torch.rand(
        tokenized_seqs.shape, generator=rng,
    ).to(tokenized_seqs.device)
    mask = (
        (probs < mask_prob)
        & (tokenized_seqs != pad_token_id)
        & (tokenized_seqs != bos_token_id)
        & (tokenized_seqs != eos_token_id)
    )
    return mask
