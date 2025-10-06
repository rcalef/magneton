import logging

import torch
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_hidden_dims(
    raw_dims: list[int | str],
    embed_dim: int,
) -> list[int]:
    """Parse a list of dimensions, possibly with special 'embed' placeholder.

    Convenience function to allow specifying 'embed' in the list of
    dimensions, corresponding to the hidden/embedding dimension of the
    underlying base model.
    """
    parsed_dims = []
    for dim in raw_dims:
        try:
            dim = int(dim)
        except ValueError:
            if dim != "embed":
                raise ValueError(f"unknown hidden dim: {dim}")
            else:
                dim = embed_dim
        parsed_dims.append(dim)
    return parsed_dims

def describe_tensor(
    x: torch.Tensor,
    prefix: str = "x",
):
    """Log summary statistics of tensor."""
    vals = x.detach().cpu().float().numpy()
    logger.info(": ".join((prefix, f"shape: {vals.shape}")))

    logger.info(": ".join((prefix, f"min: {vals.min()}")))
    logger.info(": ".join((prefix, f"max: {vals.max()}")))
    logger.info(": ".join((prefix, f"mean: {vals.mean()}")))
    logger.info(": ".join((prefix, f"std: {vals.std()}")))

    qs = np.quantile(vals, (0.25, 0.5, 0.75))
    logger.info(": ".join((prefix, f"quartiles: {qs}")))
