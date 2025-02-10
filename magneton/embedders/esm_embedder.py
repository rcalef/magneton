import huggingface_hub
import torch


from tqdm import tqdm
from typing import List, Optional
from esm.models.esm3 import ESM3
from esm.sdk import client
from esm.sdk.api import ESMProtein, ESMProteinTensor, SamplingConfig

from magneton.embedders.base_embedder import BaseEmbedder
from magneton.utils import get_chunk_idxs
#from magneton.config.base_config import ESMConfig

class ESMConfig:
    #TODO define elsewhere
    pass

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
        idxs = get_chunk_idxs(seq_len, max_len=self.config.max_seq_length)

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

    @classmethod
    def get_required_input_type(cls) -> str:
        return "sequence"