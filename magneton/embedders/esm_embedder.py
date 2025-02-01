from typing import List, Optional
import torch
from esm.models.esm3 import ESM3
from esm.sdk import client
from esm.sdk.api import ESMProtein, ESMProteinTensor, SamplingConfig
import huggingface_hub
from tqdm import tqdm

from .base_embedder import BaseEmbedder
from ..config.base_config import ESMConfig

class ESMEmbedder(BaseEmbedder):
    """ESM protein embedding model"""
    
    def __init__(self, config: ESMConfig):
        super().__init__(config)
        huggingface_hub.login()
        self.model = ESM3.from_pretrained(config.model_name, device=self.device)
        
    def process_protein(self, seq: str) -> torch.Tensor:
        """Process a single protein sequence through ESM"""
        protein = ESMProtein(sequence=seq)
        full_tensor = client.encode(protein).to("cpu")
        
        # Get chunk indices for long sequences
        seq_len = len(seq) + 2  # +2 for EOS and BOS
        idxs = self._get_chunk_idxs(seq_len, max_len=self.config.max_seq_length)
        
        outputs = []
        for start, end in idxs:
            sub_tensor = ESMProteinTensor(
                sequence=full_tensor.sequence[start:end]
            ).to(self.device)
            
            output = client.forward_and_sample(
                sub_tensor, 
                SamplingConfig(return_per_residue_embeddings=True)
            )
            outputs.append(output.per_residue_embedding.detach().cpu())
            
        return torch.cat(outputs)

    @torch.no_grad()
    def embed_one_prot(self, seq: str, max_len: Optional[int] = None) -> torch.Tensor:
        """Embed a single protein sequence"""
        max_len = max_len or self.config.max_seq_length
        return self.process_protein(seq)

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

    def _get_chunk_idxs(self, seq_len: int, max_len: int) -> List[tuple]:
        """Get indices for chunking long sequences"""
        num_pieces = (seq_len + max_len - 1) // max_len
        lo_size = seq_len // num_pieces
        hi_size = lo_size + 1
        num_hi = seq_len % num_pieces
        
        chunk_lens = [hi_size] * num_hi + [lo_size] * (num_pieces - num_hi)
        
        chunk_idxs = []
        curr = 0
        for chunk_len in chunk_lens:
            chunk_idxs.append((curr, curr + chunk_len))
            curr += chunk_len
        return chunk_idxs

    @classmethod
    def get_required_input_type(cls) -> str:
        return "sequence" 