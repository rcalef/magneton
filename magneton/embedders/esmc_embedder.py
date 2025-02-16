import os

from dataclasses import dataclass, field
from typing import List, Set, Tuple

import torch

from esm.models.esmc import ESMC
from esm.sdk.api import (
    ESMProtein,
    ESMProteinTensor,
    LogitsConfig,
)
from esm.tokenization import get_esmc_model_tokenizers
from esm.utils.sampling import _BatchedESMProteinTensor
from tqdm import tqdm

from magneton.constants import DataType
from magneton.embedders.base_embedder import BaseConfig, BaseEmbedder
from magneton.types import Protein
from magneton.utils import get_chunk_idxs

@dataclass
class ESMCConfig(BaseConfig):
    weights_path: str = field(kw_only=True)
    use_flash_attn: bool = field(kw_only=True, default=False)
    rep_layer: int = field(kw_only=True, default=35)
    max_seq_length: int = field(kw_only=True, default=2048)
    batch_size: int = 32


class ESMCEmbedder(BaseEmbedder):
    def __init__(
        self,
        config: ESMCConfig,
    ):
        super().__init__(config)

        with torch.device(self.device):
            self.model = ESMC(
                d_model=1152,
                n_heads=18,
                n_layers=36,
                tokenizer=get_esmc_model_tokenizers(),
                use_flash_attn=config.use_flash_attn,
            ).eval()

        state_dict = torch.load(
            os.path.join(
                config.weights_path,
                "data",
                "weights",
                "esmc_600m_2024_12_v0.pth",
            ),
            map_location=self.device,
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
        return (
            self.model.transformer.norm(logits_out.hidden_states)[self.rep_layer]
            .detach()
            .cpu()
        )

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
    def embed_batch(self, batch: List[Tuple[str, Protein]]) -> List[torch.Tensor]:
        """Embed a batch of protein sequences"""
        # Split out sequences we can run in batch inference, and those that
        # are too long and need to be chunked.
        short_idxs = []
        long_idxs = []
        short_seqs = []
        long_seqs = []
        for idx, (seq, _) in enumerate(batch):
            if len(seq) <= self.max_len:
                short_idxs.append(idx)
                short_seqs.append(seq)
            else:
                long_idxs.append(idx)
                long_seqs.append(seq)

        # Batch inference for short sequences
        seq_tokens = self.model._tokenize(short_seqs)
        protein_tensor = _BatchedESMProteinTensor(sequence=seq_tokens).to(self.device)
        short_embeds = self._get_embedding(protein_tensor)

        ret = [None] * len(batch)
        for i, (idx, seq) in enumerate(zip(short_idxs, short_seqs)):
            # Skip BOS and EOS tokens
            ret[idx] = short_embeds[i, 1 : len(seq) + 1, :]
        for i, (idx, seq) in enumerate(zip(long_idxs, long_seqs)):
            ret[idx] = self.embed_single_protein(seq)

        return ret

    @torch.no_grad()
    def embed_sequences(self, sequences: List[str]) -> List[torch.Tensor]:
        """Embed multiple protein sequences"""
        all_embeddings = []

        for seq in tqdm(sequences, desc="Processing sequences"):
            try:
                embedding = self.embed_one_prot(seq)
                all_embeddings.append(embedding)
            except Exception as e:
                print(f"Error processing sequence: {str(e)}")
                continue

        return all_embeddings

    @classmethod
    def get_required_input_type(cls) -> Set[DataType]:
        return {DataType.SEQ}

    @classmethod
    def model_name(cls) -> str:
        return "ESM-C"
