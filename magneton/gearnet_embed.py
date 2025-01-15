import torch
from torchdrug import datasets, transforms, data, layers, models
from torchdrug.layers import geometry
from typing import List
from tqdm import tqdm
import random

class GearNetEmbedder:
    def __init__(self, max_length: int = 350):
        # Set up the transforms
        self.truncate_transform = transforms.TruncateProtein(max_length=max_length, random=False)
        self.protein_view_transform = transforms.ProteinView(view='residue')
        self.transform = transforms.Compose([self.truncate_transform, self.protein_view_transform])
        
        # Set up the graph construction model
        self.graph_construction_model = layers.GraphConstruction(
            node_layers=[geometry.AlphaCarbonNode()],
            edge_layers=[
                geometry.SpatialEdge(radius=10.0, min_distance=5),
                geometry.KNNEdge(k=10, min_distance=5),
                geometry.SequentialEdge(max_distance=2)
            ],
            edge_feature="gearnet"
        )
        
        # Initialize GearNet model
        self.model = models.GearNet(
            input_dim=21,  # Number of amino acid types
            hidden_dims=[512, 512, 512],
            num_relation=7,
            edge_input_dim=59,
            num_angle_bin=8,
            batch_norm=True,
            concat_hidden=True,
            short_cut=True,
            readout="sum"
        )
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("ON CUDA")
            
    def process_protein(self, protein_data):
        """Process a single protein through the GearNet model"""
        # Pack the protein data
        _protein = data.Protein.pack([protein_data])
        
        # Construct the graph
        protein_ = self.graph_construction_model(_protein)
        
        # Create node features (one-hot encoded residue types)
        residue_ids = protein_.residue_type.tolist()
        node_features = torch.zeros((len(residue_ids), 21))
        for i, residue_id in enumerate(residue_ids):
            node_features[i, residue_id] = 1
            
        if torch.cuda.is_available():
            protein_ = protein_.cuda()
            node_features = node_features.cuda()
            
        # Get embeddings
        with torch.no_grad():
            output = self.model(protein_, node_features)
            embeddings = output['node_feature']
            
        return embeddings.cpu()

    @torch.no_grad()
    def embed_one_prot(self, protein_data, max_len: int = 350):
        """Embed a single protein and handle chunking if needed"""
        # If protein is shorter than max_len, process directly
        if len(protein_data.residue_type) <= max_len:
            return self.process_protein(protein_data)
            
        # Otherwise, chunk the protein and process each chunk
        chunks = []
        for i in range(0, len(protein_data.residue_type), max_len):
            chunk = protein_data[i:i + max_len]
            chunk_embedding = self.process_protein(chunk)
            chunks.append(chunk_embedding)
            
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def embed_sequences(self, protein_list: List[data.Protein], max_len: int = 350):
        """Embed multiple proteins"""
        all_embeddings = []
        
        for i, protein in enumerate(tqdm(protein_list, desc="Processing proteins")):
            try:
                embedding = self.embed_one_prot(protein, max_len)
                all_embeddings.append(embedding)
            except Exception as e:
                print(f"Error processing protein {i}: {str(e)}")
                continue
                
        return all_embeddings

    def mean_pool_random_windows(self, embeddings: torch.Tensor, window_size: int, 
                               num_windows: int, non_contiguous: bool = True):
        """
        Pool over random windows of the embeddings and return the mean of each window.
        embeddings: Tensor of shape (seq_len, embedding_dim)
        window_size: Number of residues in each window
        num_windows: Number of random windows to sample
        non_contiguous: If True, windows are non-contiguous
        """
        seq_len, embedding_dim = embeddings.shape
        pooled_embeddings = []

        for _ in range(num_windows):
            if non_contiguous:
                idxs = sorted(random.sample(range(seq_len), window_size))
            else:
                start_idx = random.randint(0, seq_len - window_size)
                idxs = list(range(start_idx, start_idx + window_size))
                
            window_embedding = embeddings[idxs].mean(dim=0)
            pooled_embeddings.append(window_embedding)

        return torch.stack(pooled_embeddings)

# Example usage
if __name__ == "__main__":
    # Initialize the embedder
    embedder = GearNetEmbedder(max_length=350)
    
    # A toy protein structure dataset
    class EnzymeCommissionToy(datasets.EnzymeCommission):
        url = "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/data/EnzymeCommission.tar.gz"
        md5 = "728e0625d1eb513fa9b7626e4d3bcf4d"
        processed_file = "enzyme_commission_toy.pkl.gz"
        test_cutoffs = [0.3, 0.4, 0.5, 0.7, 0.95]

    # Load your dataset
    dataset = EnzymeCommissionToy("~/protein-datasets/", 
                                transform=embedder.transform, 
                                atom_feature=None, 
                                bond_feature=None)
    
    # Get protein data
    proteins = [data["graph"] for data in dataset]
    
    # Get embeddings for all proteins
    embeddings = embedder.embed_sequences(proteins)
    
    # Example of using random window pooling on the first protein's embeddings
    first_protein_embed = embeddings[0]
    pooled_embeds = embedder.mean_pool_random_windows(
        first_protein_embed,
        window_size=10,
        num_windows=5,
        non_contiguous=False
    )
    
    print(f"Pooled embeddings shape: {pooled_embeds.shape}")