import torch

def pool_residue_embeddings(
    embeds: torch.Tensor,
    residue_mask: torch.Tensor,
) -> torch.Tensor:
    """Pool residue embeddings to protein embeddings.

    - embeds (torch.Tensor): tensor of residue level embeddings. Dims: (batch, sequence, embedding)
    - residue_mask (torch.Tensor): tensor indicating which positions should be pooled, 1 at all
        residue positions, 0 everywhere else. Dims: (batch, sequence)
    """
    assert embeds.ndim == 3, f"{embeds.ndim} != 3"
    assert residue_mask.ndim == 2, f"{residue_mask.ndim} != 2"
    assert embeds.shape[:2] == residue_mask.shape, f"{embeds.shape[:2]} != {residue_mask.shape}"

    # Add dummy dimension for broadcasting
    residue_mask = residue_mask.unsqueeze(-1)

    # Zero out the embeddings of non-residue tokens
    masked_embeddings = embeds * residue_mask

    # Sum the embeddings along the sequence length dimension
    summed_embeddings = torch.sum(masked_embeddings, dim=1)

    # Get the actual sequence lengths (number of non-pad tokens)
    seq_lengths = residue_mask.sum(dim=1, dtype=embeds.dtype)

    # Divide the summed embeddings by the sequence lengths
    return summed_embeddings / seq_lengths

def get_seq_mask(
    tokenized_seqs: torch.Tensor,
    ignore_tokens: torch.Tensor,
    rng: torch.Generator,
    mask_prob: float = 0.15,
) -> torch.Tensor:
    """Get mask for MLM tasks.

    Args:
        - tokenized_seqs (torch.Tensor) - tensor of input tokens.
        - ignore_tokens (torch.Tensor) - 1D tensor containing tokens
            to avoid masking (e.g. CLS, EOS, PAD).
        - rng (torch.Generator): Random number generator for reproducibility.
        - mask_prob (float): Masking probability.
    """
    probs = torch.rand(
        tokenized_seqs.shape, generator=rng,
    ).to(tokenized_seqs.device)

    mask = (
        (probs < mask_prob)
        & ~torch.isin(tokenized_seqs, ignore_tokens)
    )
    return mask
