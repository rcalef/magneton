from typing import List, Tuple

import numpy as np
import torch

def get_chunk_idxs(seq_len: int, max_len: int) -> List[Tuple[int, int]]:
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

def describe_tensor(
    x: torch.Tensor,
    prefix: str = "x",
):
    vals = x.detach().cpu().float().numpy()
    print(": ".join((prefix, f"shape: {vals.shape}")))

    print(": ".join((prefix, f"min: {vals.min()}")))
    print(": ".join((prefix, f"max: {vals.max()}")))
    print(": ".join((prefix, f"mean: {vals.mean()}")))
    print(": ".join((prefix, f"std: {vals.std()}")))

    qs = np.quantile(vals, (0.25, 0.5, 0.75))
    print(": ".join((prefix, f"quartiles: {qs}")))