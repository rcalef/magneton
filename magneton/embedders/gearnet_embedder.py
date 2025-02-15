from typing import List, Optional

import torch
from torchdrug import data, layers, models
from torchdrug.layers import geometry
from tqdm import tqdm

from magneton.embedders.base_embedder import BaseEmbedder
from magneton.config import EmbeddingConfig

class GearNetEmbedder(BaseEmbedder):
    """GearNet protein structure embedding model"""

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)

        # Set up transforms
        self.transform = self._setup_transforms()

        # Set up graph construction
        self.graph_construction_model = self._setup_graph_construction()

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

    def _setup_transforms(self):
        """Set up protein transforms"""
        from torchdrug import transforms
        truncate_transform = transforms.TruncateProtein(
            max_length=self.config.max_seq_length,
            random=False
        )
        protein_view_transform = transforms.ProteinView(view='residue')
        return transforms.Compose([truncate_transform, protein_view_transform])

    def _setup_graph_construction(self):
        """Set up graph construction model"""
        return layers.GraphConstruction(
            node_layers=[geometry.AlphaCarbonNode()],
            edge_layers=[
                geometry.SpatialEdge(radius=10.0, min_distance=5),
                geometry.KNNEdge(k=10, min_distance=5),
                geometry.SequentialEdge(max_distance=2)
            ],
            edge_feature="gearnet"
        )

    def process_protein(self, protein_data: data.Protein) -> torch.Tensor:
        """Process a single protein through GearNet"""
        # Pack the protein data
        _protein = data.Protein.pack([protein_data])

        # Construct the graph
        protein_ = self.graph_construction_model(_protein)

        # Create node features
        residue_ids = protein_.residue_type.tolist()
        node_features = torch.zeros((len(residue_ids), 21))
        for i, residue_id in enumerate(residue_ids):
            node_features[i, residue_id] = 1

        if torch.cuda.is_available():
            protein_ = protein_.to(self.device)
            node_features = node_features.to(self.device)

        # Get embeddings
        with torch.no_grad():
            output = self.model(protein_, node_features)
            embeddings = output['node_feature']

        return embeddings.cpu()

    @torch.no_grad()
    def embed_one_prot(self, protein_data: data.Protein, max_len: Optional[int] = None) -> torch.Tensor:
        """Embed a single protein structure"""
        max_len = max_len or self.config.max_seq_length

        # Handle chunking for long proteins
        if len(protein_data.residue_type) <= max_len:
            return self.process_protein(protein_data)

        chunks = []
        for i in range(0, len(protein_data.residue_type), max_len):
            chunk = protein_data[i:i + max_len]
            chunk_embedding = self.process_protein(chunk)
            chunks.append(chunk_embedding)

        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def embed_sequences(self, protein_list: List[data.Protein]) -> List[torch.Tensor]:
        """Embed multiple protein structures"""
        all_embeddings = []

        for protein in tqdm(protein_list, desc="Processing proteins"):
            try:
                embedding = self.embed_one_prot(protein)
                all_embeddings.append(embedding)
            except Exception as e:
                print(f"Error processing protein: {str(e)}")
                continue

        return all_embeddings

    @classmethod
    def get_required_input_type(cls) -> str:
        return "structure"