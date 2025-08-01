import joblib
import os
import sys

from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from esm.utils.misc import stack_variable_length_tensors
from torch_geometric.data import Batch
from torch_scatter import scatter_mean
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm

PROSST_REPO_PATH = (
    Path(__file__).parent.parent /
    "external" /
    "ProSST"
)
sys.path.append(str(PROSST_REPO_PATH))
from prosst.structure.encoder.gvp import AutoGraphEncoder
from prosst.structure.get_sst_seq import (
    SSTPredictor,
    process_pdb_file,
)

from magneton.config import DataConfig, TrainingConfig
from magneton.data.meta_dataset import MetaDataset
from magneton.data.substructure import LabeledSubstructure, SubstructureBatch
from magneton.embedders.base_embedder import BaseConfig, BaseDataModule, BaseEmbedder
from magneton.types import DataType
from magneton.utils import get_chunk_idxs, move_inputs_to_device

def move_prosst_inputs_to_device(
    data: torch.Tensor | Any,
    device: torch.device,
) -> torch.Tensor |  Any:
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)(
            {k: move_inputs_to_device(v, device) for k, v in data.items()}
        )
    elif isinstance(data, (tuple, list)):
        return type(data)(move_inputs_to_device(v, device) for v in data)
    elif isinstance(data, (torch.Tensor, Batch)):
        return data.to(device=device)
    return data


@dataclass
class ProSSTDataElem:
    pdb_subgraphs: list[dict] # return types from ProSST's `process_pdb_file` function
    substructures: list[LabeledSubstructure]
    sequence: torch.Tensor
    prot_id: str
    protein_length: int


@dataclass
class ProSSTBatch(SubstructureBatch):
    batch_graphs: Batch
    tokenized_seqs: torch.Tensor
    protein_lengths: list[int]

    def to(self, device: str):
        super().to(device)
        self.batch_graphs = self.batch_graphs.to(device)
        self.tokenized_seqs = self.tokenized_seqs.to(device)
        return self


class ProSSTDataSet(MetaDataset):
    def __init__(
        self,
        data_config: DataConfig,
    ):
        prosst_params = data_config.model_specific_params
        self.subgraph_depth = prosst_params.get("subgraph_depth", None)
        self.max_distance = prosst_params.get("max_distance", 10)
        self.num_threads = prosst_params.get("num_threads", 1)

        model_path = prosst_params["model_path"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        super().__init__(
            data_config=data_config,
            want_datatypes=[DataType.STRUCT, DataType.SUBSTRUCT],
            load_fasta_in_mem=True,
        )

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx: int) -> ProSSTDataElem:
        # elem: BatchElem
        elem = self._prot_to_elem(self.dataset[idx])
        prot_id = elem.protein_id

        pdb_subgraphs, result_dict, node_count = process_pdb_file(
            elem.structure_path,
            self.subgraph_depth,
            self.max_distance,
            self.num_threads,
            None,
        )
        if pdb_subgraphs is None:
            raise ValueError(
                f"ProSST: error parsing PDB file for {prot_id} ({elem.structure_path}): {result_dict['error']}"
            )

        tokenized_seq = self.tokenizer(
            result_dict["aa_seq"],
            return_tensors="pt",
        )["input_ids"].squeeze()
        return ProSSTDataElem(
            pdb_subgraphs=pdb_subgraphs,
            substructures=elem.substructures,
            sequence=tokenized_seq,
            prot_id=prot_id,
            protein_length=len(result_dict["aa_seq"]),
        )


def prosst_collate(
    entries: list[ProSSTDataElem],
    pad_id: int,
) -> ProSSTBatch:
    """
    Collate the entries into a batch.
    """
    all_pdb_subgraphs = []
    substructs = []
    seqs = []
    lens = []
    prot_ids = []
    for e in entries:
        if len(e.substructures) == 0:
            raise ValueError(
                f"empty substructures should be filtered upstream: {entries}"
            )
        all_pdb_subgraphs.extend(e.pdb_subgraphs)
        substructs.append(e.substructures)
        prot_ids.append(e.prot_id)
        seqs.append(e.sequence)
        lens.append(e.protein_length)

    # Below copied from prosst.structure.get_sst_seq.py:324-325
    batch_graphs = Batch.from_data_list(all_pdb_subgraphs)
    batch_graphs.node_s = torch.zeros_like(batch_graphs.node_s)
    padded_tensor = stack_variable_length_tensors(
        seqs,
        constant_value=pad_id,
    )

    return ProSSTBatch(
        batch_graphs=batch_graphs,
        substructures=substructs,
        tokenized_seqs=padded_tensor,
        protein_lengths=lens,
        prot_ids=prot_ids,
    )


class ProSSTDataModule(BaseDataModule):
    def __init__(
        self,
        data_config: DataConfig,
        train_config: TrainingConfig,
    ):
        super().__init__(data_config, train_config)

    def _get_split_info(self, split: str) -> tuple[str, str]:
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
        dataset = ProSSTDataSet(
            data_config=config,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            collate_fn=partial(prosst_collate, pad_id=dataset.tokenizer.pad_token_id),
            num_workers=0,
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
class ProSSTConfig(BaseConfig):
    weights_path: str = field(kw_only=True)
    structure_vocab_size: int = field(kw_only=True, default=2048)
    max_distance: int = field(kw_only=True, default=10)
    max_batch_nodes: int = field(kw_only=True, default=10000)
    num_pdb_procs: int = field(kw_only=True, default=16)
    # Default to final layer hidden states
    rep_layer: int = field(kw_only=True, default=12)


class ProSSTEmbedder(BaseEmbedder):
    def __init__(
        self,
        config: ProSSTConfig,
        frozen: bool = True,
    ):
        super().__init__(config)

        self.vocab_size = config.structure_vocab_size
        self.max_distance = config.max_distance
        self.max_batch_ndes = config.max_batch_nodes
        self.num_pdb_procs = config.num_pdb_procs
        self.rep_layer = config.rep_layer

        # Below copied from prosst.structure.get_sst_seq.py:396-411
        static_path = (
            PROSST_REPO_PATH /
            "prosst" /
            "structure" /
            "static"
        )
        graph_encoder_model_path = str(static_path / "AE.pt")
        node_dim = (256, 32)
        edge_dim = (64, 2)
        graph_encoder_model = AutoGraphEncoder(
            node_in_dim=(20, 3),
            node_h_dim=node_dim,
            edge_in_dim=(32, 1),
            edge_h_dim=edge_dim,
            num_layers=6,
        )
        graph_encoder_model.load_state_dict(torch.load(graph_encoder_model_path))
        self.encoder_model = graph_encoder_model
        self.cluster_model = joblib.load(str(static_path / f"{self.vocab_size}.joblib"))

        self.model = AutoModelForMaskedLM.from_pretrained(config.weights_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(config.weights_path, trust_remote_code=True)
        self.embed_dim = self.model.get_output_embeddings().in_features

        if frozen:
            self.encoder_model = self.encoder_model.eval()
            self.model = self.model.eval()
            self._freeze()
        # Freeze everything after the layer we're
        # using for extracting representations to
        # avoid DDP errors.
        else:
            self._unfreeze(unfreeze_all=False)

        # For masking when calculating original MLM loss
        self.rng = torch.Generator().manual_seed(42)

    def _freeze(self):
        for params in self.model.parameters():
            params.requires_grad = False
        for params in self.encoder_model.parameters():
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

    @torch.no_grad()
    def _get_structure_toks_batch(self, batch: ProSSTBatch) -> list[dict]:
        h_V = (batch.batch_graphs.node_s, batch.batch_graphs.node_v)
        h_E = (batch.batch_graphs.edge_s, batch.batch_graphs.edge_v)
        node_emebddings = self.encoder_model.get_embedding(
            h_V,
            batch.batch_graphs.edge_index,
            h_E,
        )
        graph_emebddings = scatter_mean(node_emebddings, batch.batch_graphs.batch, dim=0).cpu()
        norm_graph_emebddings = F.normalize(graph_emebddings, p=2, dim=1)
        batch_structure_labels = self.cluster_model.predict(
                norm_graph_emebddings
        ) + 3
        # +3 here is taken from ProSST author's notebook for variant effect prediction,
        # see cell 6 here:
        #  https://github.com/ai4protein/ProSST/blob/main/zero_shot/score_mutant.ipynb
        batch_structure_labels = batch_structure_labels.tolist()

        start, end = 0, 0
        structure_toks = []
        for seq_len in batch.protein_lengths:
            end += seq_len
            # Similar to above, prepending 1 and appending 2 tokens from same notebook
            this_prot_toks = [1] + batch_structure_labels[start:end] + [2]
            structure_toks.append(torch.tensor(
                this_prot_toks,
                device=node_emebddings.device,
                dtype=torch.int,
            ))
            start = end

        structure_toks = stack_variable_length_tensors(
            structure_toks,
            constant_value=self.tokenizer.pad_token_id,
        )
        return structure_toks

    def _get_embedding(
        self,
        protein_tensor: torch.Tensor,
    ) -> torch.Tensor:
        logits_out = self.model.forward(protein_tensor)

        return self.model.transformer.norm(logits_out.hidden_states[self.rep_layer])

    def embed_batch(self, batch: ProSSTBatch) -> torch.Tensor:
        """Embed a batch of pre-tokenized protein sequences"""
        structure_tokens = self._get_structure_toks_batch(batch)
        attention_mask = torch.ones_like(batch.tokenized_seqs)
        attention_mask[batch.tokenized_seqs == self.tokenizer.pad_token_id] = 0

        out = self.model(
            input_ids=batch.tokenized_seqs,
            ss_input_ids=structure_tokens,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Normalize along the embedding dimension
        return F.normalize(out.hidden_states[self.rep_layer], dim=-1)

    # the following two functions are deprecated for the current data module setup
    @torch.no_grad()
    def embed_single_protein(self, seq: str) -> torch.Tensor:
        """Process a single protein sequence through ESM"""
        pass

    @torch.no_grad()
    def embed_sequences(self, sequences: list[str]) -> list[torch.Tensor]:
        """Embed multiple protein sequences"""
        pass

    def calc_original_loss(
        self,
        batch: ProSSTBatch,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """NOTE: this modifies the original tokenized seq tensor, call last."""
        return 0

    def get_embed_dim(self):
        return self.embed_dim

    def model_name(self) -> str:
        return f"ESM-C_{self.model_size}"

    @classmethod
    def get_required_input_type(cls) -> set[DataType]:
        return {DataType.SEQ}
