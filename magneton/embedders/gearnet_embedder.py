import os

from dataclasses import dataclass, field, replace
from functools import partial
from typing import List, Set, Tuple

import torch
from torchdrug import data, layers, models, transforms
from torchdrug.layers import geometry
from torchdrug.data import Protein, PackedProtein
import re
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

@dataclass
class GearNetDataElem:
    protein_id: str
    structure_path: str
    substructures: List[LabeledSubstructure]


@dataclass
class GearNetBatch(SubstructureBatch):
    protein_ids: List[str]
    packed_protein: PackedProtein

    def to(self, device: str):
        super().to(device)
        self.packed_protein = self.packed_protein.to(device)
        return self

def gearnet_collate(
    entries: List[GearNetDataElem],
    pad_id: int,
    drop_empty_substructures: bool = True,
) -> GearNetBatch:
    """
    Collate the data elements into a batch.
    Should act on packed proteins
    """
    if drop_empty_substructures:
        entries = [e for e in entries if len(e.substructures) > 0]

    # padded_tensor = stack_variable_length_tensors(
    #     [x.tokenized_seq for x in entries],
    #     constant_value=pad_id,
    # )
    # substructs = [x.substructures for x in entries]
    # return GearNetBatch(
    #     tokenized_seq=_BatchedESMProteinTensor(sequence=padded_tensor),
    #     substructures=substructs,
    # )

    # Load and pack proteins
    proteins = []
    for entry in entries:
        try:
            prot = Protein.from_pdb(entry.structure_path, atom_feature="position")
            proteins.append(prot)
        except Exception as e:
            print(f"Error loading protein {entry.protein_id}: {str(e)}")
            continue

    packed_protein = Protein.pack(proteins)

    return GearNetBatch(
        protein_ids=[x.protein_id for x in entries],
        packed_protein=packed_protein,
        substructures=[x.substructures for x in entries]
    )

class GearNetDataSet(MetaDataset):
    def __init__(
        self,
        data_config: DataConfig,
    ):
        super().__init__(
            data_config=data_config,
            want_datatypes=[DataType.STRUCT, DataType.SUBSTRUCT],
            load_fasta_in_mem=True,
        )
        # self.tokenizer = get_esmc_tokenizers()
        self.pdb_template = "/weka/scratch/weka/kellislab/rcalef/data/pdb_alphafolddb/AF-%s-F1-model_v4.pdb"

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx: int) -> GearNetDataElem:
        elem = self._prot_to_elem(self.dataset[idx])
        uniprot_id = re.sub(r'\|.*', '', elem.protein_id)
        return GearNetDataElem(
            protein_id=elem.protein_id,
            structure_path=self.pdb_template % uniprot_id,
            substructures=elem.substructures
        )

class GearNetDataModule(BaseDataModule):
    def __init__(
        self,
        data_config: DataConfig,
        train_config: TrainingConfig,
    ):
        super().__init__(data_config, train_config)
        self.config = data_config
        self.batch_size = train_config.batch_size

    def _get_loader(self, dataset: GearNetDataSet) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

    def _get_split_info(self, split: str) -> Tuple[str, str]:
        if split == "all":
            return self.config.data_dir, self.config.prefix
        else:
            return (
                os.path.join(
                    self.config.data_dir, "dataset_splits", "seq_splits", f"{split}_sharded"
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
            self.config,
            data_dir=data_dir,
            prefix=prefix,
        )
        dataset = GearNetDataSet(
            data_config=config,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=gearnet_collate,
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
class GearNetConfig(BaseConfig):
    weights_path: str = field(kw_only=True)
    max_seq_length: int = field(kw_only=True, default=2048)
    hidden_dims: list = field(kw_only=True, default_factory=lambda: [512, 512, 512])
    num_relation: int = field(kw_only=True, default=7)
    edge_input_dim: int = field(kw_only=True, default=59)
    num_angle_bin: int = field(kw_only=True, default=8)

class GearNetEmbedder(BaseEmbedder):
    """GearNet protein structure embedding model"""

    def __init__(
            self,
            config: GearNetConfig,
            frozen: bool = True,
    ):
        super().__init__(config)

        self.graph_construction_model = layers.GraphConstruction(
            node_layers=[geometry.AlphaCarbonNode()],
            edge_layers=[
                geometry.SequentialEdge(max_distance=2),
                geometry.SpatialEdge(radius=10.0, min_distance=5),
                geometry.KNNEdge(k=10, min_distance=5)
            ],
            edge_feature="gearnet"
        )

        # Initialize GearNet model
        self.model = models.GearNet(
            input_dim=21,
            hidden_dims=config.hidden_dims,
            num_relation=config.num_relation,
            edge_input_dim=config.edge_input_dim,
            num_angle_bin=config.num_angle_bin,
            batch_norm=True,
            concat_hidden=True,
            short_cut=True,
            readout="sum"
        ).to(self.device)

        if frozen:
            for param in self.parameters():
                param.requires_grad = False

        if config.weights_path:
            state_dict = torch.load(config.weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

    @torch.no_grad()
    def _get_embedding(self, protein: PackedProtein) -> torch.Tensor:
        """Get embeddings from a packed protein"""
        # Construct graph
        protein = self.graph_construction_model(protein)

        # Create one-hot encoded node features
        node_features = torch.zeros((len(protein.residue_type), 21), device=self.device)
        for i, residue_id in enumerate(protein.residue_type):
            node_features[i, residue_id] = 1

        # Get embeddings
        output = self.model(protein, node_features)
        return output['node_feature']

    @torch.no_grad()
    def embed_batch(self, batch: GearNetBatch) -> List[torch.Tensor]:
        """Embed a batch of proteins"""
        # embeddings = self._get_embedding(batch.packed_protein)

        # # Split embeddings back into individual proteins
        # protein_lengths = batch.packed_protein.num_residues
        # return torch.split(embeddings, protein_lengths.tolist())

    # TODO Just pass in the packed protein from the dataloader
    @torch.no_grad()
    def embed_single_structure(self, structure_path: str) -> torch.Tensor:
        """Embed a single protein structure"""
        structure = Protein.from_pdb(structure_path, atom_feature="position")
        packed_protein = Protein.pack([structure]).to(self.device)
        embedding = self._get_embedding(packed_protein)
        return embedding

    @torch.no_grad()
    def embed_structures(self, structure_list: List[str]) -> List[torch.Tensor]:
        """Embed multiple protein structures"""
        all_embeddings = []

        for structure_path in tqdm(structure_list, desc="Processing protein structures"):
            try:
                structure = Protein.from_pdb(structure_path, atom_feature="position")
                packed_protein = Protein.pack([structure]).to(self.device)
                embedding = self._get_embedding(packed_protein)
                all_embeddings.append(embedding)
            except Exception as e:
                print(f"Error processing structure: {str(e)}")
                continue

        return all_embeddings

    def get_embed_dim(self):
        return self.model.output_dim

    @classmethod
    def get_required_input_type(cls) -> Set[DataType]:
        return {DataType.STRUCT}

    @classmethod
    def model_name(cls) -> str:
        return "GearNet"