from magneton.io.interpro import parse_from_pkl_w_fasta
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import torch
from typing import List, Tuple
import numpy as np
import os

from magneton.esm_embed import embed_sequences, embed_one_prot
from esm.models.esm3 import ESM3
import huggingface_hub

pkl_path = "/rdma/vast-rdma/vast-home/rcalef/transfer/parse_swissprot.pkl"
fasta_path = "/rdma/vast-rdma/vast-home/rcalef/transfer/uniprot_sprot.fasta.bgz"

prots_w_seq = []
for i, prot in enumerate(parse_from_pkl_w_fasta(pkl_path, fasta_path)):
    prots_w_seq.append(prot)
# print(prots_w_seq[0][0].uniprot_id)
# prots_w_seq[0][0].print()

def get_unique_element_names(prot_list: List[Tuple]) -> List[str]:
    """Extract all unique element names from the protein list."""
    element_names = set()
    for prot_tuple in prot_list:
        protein = prot_tuple[0]
        for entry in protein.entries:
            element_names.add(entry.element_name)
    return sorted(list(element_names))

# Get unique element names and create label mapping
unique_elements = get_unique_element_names(prots_w_seq[:500])
element_to_idx = {elem: idx for idx, elem in enumerate(unique_elements)}

print(f"Found {len(unique_elements)} unique element names")

# Save the label mapping
save_dir = "saved_embeddings"
os.makedirs(save_dir, exist_ok=True)
np.save(os.path.join(save_dir, "element_names.npy"), np.array(unique_elements))

def get_protein_sequence(uniprot_id: str) -> Tuple[str, str]:
    """Fetch protein sequence from UniProt API.

    Args:
        uniprot_id: UniProt identifier
    Returns:
        Tuple of (uniprot_id, sequence or error message)
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    try:
        response = requests.get(url)
        response.raise_for_status()
        sequence = ''.join(response.text.split('\n')[1:])  # Skip header and join lines
        return uniprot_id, sequence
    except Exception as e:
        return uniprot_id, f"Error: {str(e)}"

def get_protein_sequences(prot_list: List, max_workers: int = 10, batch_size: int = 50) -> List[str]:
    """Fetch multiple protein sequences in parallel.

    Args:
        prot_list: List of protein tuples
        max_workers: Number of parallel threads
        batch_size: Number of proteins to process in each batch
    Returns:
        List of successfully retrieved sequences
    """
    sequences = []
    failed_ids = []

    # Extract all UniProt IDs
    uniprot_ids = [prot_tuple[0].uniprot_id for prot_tuple in prot_list]

    # Process in batches to avoid overwhelming the API
    for i in range(0, len(uniprot_ids), batch_size):
        batch = uniprot_ids[i:i + batch_size]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all requests for this batch
            future_to_id = {
                executor.submit(get_protein_sequence, uniprot_id): uniprot_id
                for uniprot_id in batch
            }

            # Process completed requests with progress bar
            for future in tqdm(as_completed(future_to_id),
                             total=len(batch),
                             desc=f"Batch {i//batch_size + 1}/{len(uniprot_ids)//batch_size + 1}"):
                uniprot_id, result = future.result()
                if not result.startswith('Error'):
                    sequences.append(result)
                else:
                    failed_ids.append(uniprot_id)
                    print(f"\nFailed to fetch {uniprot_id}: {result}")

        # Add a small delay between batches to be nice to the API
        time.sleep(1)

    print(f"\nRetrieved {len(sequences)} sequences successfully")
    print(f"Failed to retrieve {len(failed_ids)} sequences")

    return sequences

# Get all sequences
all_sequences = get_protein_sequences(prots_w_seq[:500])
print(f"Successfully retrieved {len(all_sequences)} sequences")

huggingface_hub.login()

client =  ESM3.from_pretrained("esm3_sm_open_v1", device=torch.device("cuda"))

# Generate embeddings and labels
all_embeddings = []
all_labels = []

def create_position_embedding(sequence_embedding: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Create embedding for a specific position range by mean pooling."""
    # Adjust indices to account for sequence embedding shape
    start_idx = min(start, sequence_embedding.shape[0]-1)
    end_idx = min(end, sequence_embedding.shape[0])
    return sequence_embedding[start_idx:end_idx].mean(dim=0)

for seq, prot_tuple in zip(all_sequences, prots_w_seq[:500]):
    protein = prot_tuple[0]
    try:
        # Get sequence embedding
        sequence_embedding = embed_one_prot(seq, client)

        # Create one embedding and label for each InterPro entry
        for entry in protein.entries:
            for start, end in entry.positions:
                # Create embedding for this position
                pos_embedding = create_position_embedding(sequence_embedding, start, end)
                all_embeddings.append(pos_embedding)

                # Create one-hot label
                label_idx = element_to_idx[entry.element_name]
                all_labels.append(label_idx)

    except Exception as e:
        print(f"Error processing protein {protein.uniprot_id}: {str(e)}")
        continue

# Convert to numpy arrays
embeddings_array = torch.stack(all_embeddings).cpu().numpy()
labels_array = np.array(all_labels)

# Save embeddings and labels
np.save(os.path.join(save_dir, "protein_embeddings.npy"), embeddings_array)
np.save(os.path.join(save_dir, "protein_labels.npy"), labels_array)

print(f"Saved embeddings with shape: {embeddings_array.shape}")
print(f"Saved labels with shape: {labels_array.shape}")