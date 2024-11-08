# %set_env TOKENIZERS_PARALLELISM=false
import numpy as np
import torch
import py3Dmol
import huggingface_hub
import requests
import random
from tqdm import tqdm

from esm.utils.structure.protein_chain import ProteinChain
from esm.models.esm3 import ESM3
from esm.sdk import client
from esm.sdk.api import (
    ESMProtein,
    GenerationConfig,
    ESMProteinTensor, 
    SamplingConfig
)

from typing import List

def get_chunk_idxs(
    seq_len: int,
    max_len: int,
) -> List[int]:
    num_pieces = (seq_len + max_len - 1) // max_len
    lo_size = seq_len // num_pieces
    hi_size = lo_size + 1
    num_hi = seq_len % num_pieces

    chunk_lens = [hi_size for _ in range(num_hi)] + [lo_size for _ in range(num_pieces - num_hi)]

    chunk_idxs = []
    curr = 0
    for chunk_len in chunk_lens:
        chunk_idxs.append((curr, curr+chunk_len))
        curr += chunk_len
    return chunk_idxs

@torch.no_grad()
def embed_one_prot(seq: str, client, max_len: int = 2048):
    seq_len = len(seq)
    # + 2 for EOS and BOS
    idxs = get_chunk_idxs(seq_len+2, max_len=max_len)
    protein = ESMProtein(
        sequence=seq,
    )
    full_tensor = client.encode(protein).to("cpu")

    outputs = []
    for (start, end) in idxs:
        sub_tensor = ESMProteinTensor(
            sequence=full_tensor.sequence[start:end]
        ).to("cuda")

        output = client.forward_and_sample(
            sub_tensor, SamplingConfig(return_per_residue_embeddings=True)
        )
        outputs.append(output.per_residue_embedding.detach().cpu())
    outputs = torch.cat(outputs)
    # print(outputs.shape)
    # return outputs[1:-1,:].mean(dim=0)
    return outputs

@torch.no_grad()
def embed_sequences(sequences: List[str], client, max_len: int = 2048):
    """
    Embed multiple protein sequences and return their embeddings
    
    Args:
        sequences: List of protein sequences
        client: ESM3 model client
        max_len: Maximum sequence length to process at once
    
    Returns:
        List of embeddings tensors
    """
    all_embeddings = []
    
    for i, seq in enumerate(tqdm(sequences, desc="Processing sequences")):
        try:
            embedding = embed_one_prot(seq, client, max_len)
            all_embeddings.append(embedding)
        except Exception as e:
            print(f"Error processing sequence {i}: {str(e)}")
            continue
            
    return all_embeddings

def get_protein_sequence(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    response = requests.get(url)
    
    if response.status_code == 200:
        # Parse the FASTA format to extract only the sequence
        fasta_data = response.text
        sequence = ''.join(fasta_data.split('\n')[1:])  # Skip header and join lines
        return sequence
    else:
        return f"Error: Unable to retrieve data (status code {response.status_code})"

def get_random_window_idxs(seq_len: int, window_size: int, non_contiguous: bool = True) -> List[int]:
    """
    Get random indices for a window over the sequence. If non_contiguous is True,
    the indices will not be contiguous.
    """
    if non_contiguous:
        print(sorted(random.sample(range(seq_len), window_size)))
        return sorted(random.sample(range(seq_len), window_size))
    else:
        start_idx = random.randint(0, seq_len - window_size)
        print(list(range(start_idx, start_idx + window_size)))
        return list(range(start_idx, start_idx + window_size))

@torch.no_grad()
def mean_pool_random_windows(embeddings: torch.Tensor, window_size: int, num_windows: int, non_contiguous: bool = True):
    """
    Pool over random windows of the embeddings and return the mean of each window.
    embeddings: Tensor of shape (seq_len, embedding_dim)
    window_size: Number of amino acids in each window
    num_windows: Number of random windows to sample
    non_contiguous: If True, windows are non-contiguous
    """
    seq_len, embedding_dim = embeddings.shape
    pooled_embeddings = []

    for _ in range(num_windows):
        idxs = get_random_window_idxs(seq_len, window_size, non_contiguous=non_contiguous)
        window_embedding = embeddings[idxs].mean(dim=0)
        pooled_embeddings.append(window_embedding)

    return torch.stack(pooled_embeddings)

# # Assume 'embed' contains the per-amino acid embeddings from the previous steps
# window_size = 10  # Size of the random window
# num_windows = 5   # Number of random windows to sample

# # Get the mean-pooled embeddings for random windows
# pooled_embeds = mean_pool_random_windows(embed, window_size=window_size, num_windows=num_windows, non_contiguous=False)

# print(pooled_embeds.shape)  # Shape will be (num_windows, embedding_dim)