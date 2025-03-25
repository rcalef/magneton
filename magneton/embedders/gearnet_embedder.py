import os

from dataclasses import dataclass, field, replace
from functools import partial
from typing import List, Set, Tuple

import torch
from torchdrug import data, layers, models, transforms
from torchdrug.layers import geometry
from torchdrug.data import Protein as torchProtein
from rdkit import Chem
from torchdrug.data import PackedProtein as torchPackedProtein
import re
from tqdm import tqdm

from magneton.config import DataConfig, TrainingConfig
from magneton.data.meta_dataset import MetaDataset
from magneton.data.substructure import LabeledSubstructure
from magneton.embedders.base_embedder import BaseConfig, BaseDataModule, BaseEmbedder
from magneton.types import DataType
from magneton.utils import get_chunk_idxs

class CustomProtein(torchProtein):
    @classmethod
    def from_pdb(cls, pdb_file, atom_feature="default", bond_feature="default", residue_feature="default",
                 mol_feature=None, kekulize=False):
        """
        Create a protein from a PDB file.

        Parameters:
            pdb_file (str): file name
            atom_feature (str or list of str, optional): atom features to extract
            bond_feature (str or list of str, optional): bond features to extract
            residue_feature (str, list of str, optional): residue features to extract
            mol_feature (str or list of str, optional): molecule features to extract
            kekulize (bool, optional): convert aromatic bonds to single/double bonds.
                Note this only affects the relation in ``edge_list``.
                For ``bond_type``, aromatic bonds are always stored explicitly.
                By default, aromatic bonds are stored.
        """
        if not os.path.exists(pdb_file):
            raise FileNotFoundError("No such file `%s`" % pdb_file)
        mol = Chem.MolFromPDBFile(pdb_file, sanitize=False)
        if mol is None:
            raise ValueError("RDKit cannot read PDB file `%s`" % pdb_file)
        return cls.from_molecule(mol, atom_feature, bond_feature, residue_feature, mol_feature, kekulize)

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
class GearNetDataElem:
    protein_id: str
    structure: torchProtein
    substructures: List[LabeledSubstructure]

@dataclass
class GearNetBatch(SubstructureBatch):
    protein_ids: List[str]
    packed_protein: torchPackedProtein

    def to(self, device: str):
        super().to(device)
        self.packed_protein = self.packed_protein.to(device)
        return self

def gearnet_collate(
    entries: List[GearNetDataElem],
    drop_empty_substructures: bool = True,
) -> GearNetBatch:
    """
    Collate the data elements into a batch.
    Should act on packed proteins.
    Should run single threaded and not do any processing.
    """

    if drop_empty_substructures:
        entries = [e for e in entries if len(e.substructures) > 0]
    
    # Load and pack proteins
    proteins = [entry.structure for entry in entries]
    # for entry in entries:
    #     try:
    #         prot = Protein.from_pdb(entry.structure_path, atom_feature="position")
    #         proteins.append(prot)
    #     except Exception as e:
    #         print(f"Error loading protein {entry.protein_id}: {str(e)}")
    #         continue
    
    packed_protein = torchProtein.pack(proteins)
    # print(f"Packed Protein: {packed_protein}")
    
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
        self.pdb_template = data_config.struct_template

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx: int) -> GearNetDataElem:
        elem = self._prot_to_elem(self.dataset[idx])
        uniprot_id = re.sub(r'\|.*', '', elem.protein_id)
        # instead of structure path, return from_pdb so that we can run this in parallel
        return GearNetDataElem(
            protein_id=elem.protein_id,
            structure= CustomProtein.from_pdb(self.pdb_template % uniprot_id, atom_feature="position"),
            substructures=elem.substructures,
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
            batch_size=self.batch_size,
            shuffle=True,
        )

    def _get_split_info(self, split: str) -> Tuple[str, str]:
        if split == "all":
            return self.config.data_dir, self.config.prefix
        else:
            return (
                os.path.join(
                    self.config.data_dir, f"{split}_sharded"
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
    hidden_dims: list = field(kw_only=True, default_factory=lambda: [512] * 6)
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
        ).eval()

        # print(self.model)

        if frozen:
            for param in self.parameters():
                param.requires_grad = False

        if config.weights_path:
            state_dict = torch.load(config.weights_path)
            self.model.load_state_dict(state_dict)

        # if config.weights_path:
        #     # Load and inspect the state dict before creating the model
        #     state_dict = torch.load(config.weights_path)
        #     print("Model architecture from weights:")
        #     # Print keys that contain 'layers' to understand the structure
        #     layer_keys = [k for k in state_dict.keys() if 'layers' in k]
        #     for k in sorted(layer_keys):
        #         print(f"Layer: {k} -> Shape: {state_dict[k].shape}")

    @torch.no_grad()
    def _get_embedding(self, protein: torchPackedProtein) -> torch.Tensor:
        """Get embeddings from a packed protein"""
        # print(protein)
        # # Construct graph
        # protein = self.graph_construction_model(protein)

        # # Create one-hot encoded node features
        # node_features = torch.zeros((len(protein.residue_type), 21))
        # for i, residue_id in enumerate(protein.residue_type):
        #     node_features[i, residue_id] = 1

        # # Get embeddings
        # output = self.model(protein, node_features)

        # # TODO Check shape of this and make sure it's the right tensor shape and not a graph
        # # TODO Check ordering and amino acid correspondence of embeddings
        # return output['node_feature']

        # Construct graph
        protein = self.graph_construction_model(protein)

        # Create one-hot encoded node features
        node_features = torch.zeros((len(protein.residue_type), 21))
        for i, residue_id in enumerate(protein.residue_type):
            node_features[i, residue_id] = 1

        # Device handling
        device = self.model.device
        protein = protein.to(device)
        node_features = node_features.to(device)

        # Get embeddings
        output = self.model(protein, node_features)
        node_embeddings = output['node_feature']  # Shape: (total_nodes, embedding_dim)
        # print(node_embeddings.shape)

        # Get the number of residues per protein in the batch
        num_residues = protein.num_residues  # List of residue counts for each protein
        batch_size = len(num_residues)
        max_len = max(num_residues)
        embedding_dim = node_embeddings.shape[1]
        # print(num_residues, batch_size, max_len, embedding_dim)

        # Create padded output tensor
        padded_embeddings = torch.zeros(
            (batch_size, max_len, embedding_dim),
            device=node_embeddings.device
        )

        # Fill in the embeddings for each protein
        start_idx = 0
        for i, length in enumerate(num_residues):
            padded_embeddings[i, :length] = node_embeddings[start_idx:start_idx + length]
            start_idx += length

        return padded_embeddings

    @torch.no_grad()
    def embed_batch(self, batch: GearNetBatch) -> torch.Tensor:
        """Embed a batch of proteins"""
        # TODO Check if there's an additional operation that needs to happen after above
        return self._get_embedding(batch.packed_protein)

    # Note: the following two functions are not used in the MLP training runs, device issues
    @torch.no_grad()
    def embed_single_protein(self, structure_path: str) -> torch.Tensor:
        """Embed a single protein structure"""
        structure = torchProtein.from_pdb(structure_path, atom_feature="position")
        packed_protein = torchProtein.pack([structure])
        embedding = self._get_embedding(packed_protein)
        return embedding

    @torch.no_grad()
    def embed_structures(self, structure_list: List[str]) -> List[torch.Tensor]:
        """Embed multiple protein structures"""
        all_embeddings = []

        for structure_path in tqdm(structure_list, desc="Processing protein structures"):
            try:
                structure = torchProtein.from_pdb(structure_path, atom_feature="position")
                packed_protein = torchProtein.pack([structure])
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