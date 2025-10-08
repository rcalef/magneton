import os
from pathlib import Path

import torch.distributed as dist

def should_run_single_process() -> bool:
    no_distributed = not dist.is_initialized()
    return no_distributed or dist.get_rank() == 0

DATA_DIR_ENV_VAR = "MAGNETON_DATA_DIR"
def get_data_dir() -> Path:
    data_dir = Path(os.environ.get(DATA_DIR_ENV_VAR))
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Magneton data directory not found at: {str(data_dir)}, please "
            f"set the '{DATA_DIR_ENV_VAR}' environment variable or download "
            "the dataset from HuggingFace: https://huggingface.co/datasets/rcalef/magneton-data"
        )
    return data_dir
