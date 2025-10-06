import torch.distributed as dist

def should_run_single_process() -> bool:
    no_distributed = not dist.is_initialized()
    return no_distributed or dist.get_rank() == 0
