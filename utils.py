import importlib
import os
import logging
import inspect
from datetime import timedelta
from typing import Any

from rich.logging import RichHandler
import torch
from torch.distributed import get_rank, is_initialized


def init_distributed_environment(args):
    from vllm_experience_batcher import get_or_create_experience_batcher
    import ray
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=180))
    args.global_rank = torch.distributed.get_rank()
    tensor = torch.ByteTensor([False]).cuda()
    torch.distributed.all_reduce(tensor)
    torch.distributed.barrier()
    
    ray.init(address="auto", namespace="test")
    args.batcher_actor = get_or_create_experience_batcher(args.experience_batcher_name)
    args.actor_registry = ray.get_actor("generation_vllm_registry")
    args.reference_registry = ray.get_actor("logprob_vllm_registry")
    ray.get(args.batcher_actor.register_training_process.remote(args.global_rank, args.max_tokens_per_gpu))
    torch.distributed.barrier()

def setup_logger(level="DEBUG"):
    logging.basicConfig(
        level=level, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )

def get_caller(num_frames=1):
    frame = inspect.currentframe().f_back
    for _ in range(num_frames - 1):
        frame = frame.f_back
    file_name = frame.f_code.co_filename
    line_number = frame.f_lineno
    return f"In {file_name}, line {line_number}"

def log_rank_0(msg, include_caller=False, rank=None, to_print=False):
    if rank is None:
        rank = get_rank() if is_initialized() else 0
    if rank <= 0:
        if include_caller:
            msg = f"{get_caller(num_frames=2)}: {msg}"
        if to_print:
            print(msg)
        else:
            logging.info(msg)

def patch_target_module(
    to_patch: str,
    replace_with: Any,
):
    to_patch = to_patch.split(".")
    assert len(to_patch) > 1, "must have an object to patch"

    to_patch, obj_name_to_patch = to_patch[:-1], to_patch[-1]
    to_patch = ".".join(to_patch)
    source = importlib.import_module(to_patch)
    setattr(source, obj_name_to_patch, replace_with)

# Simple namespace class to mimic argparse.Namespace behavior
class ArgsNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        items = (f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{type(self).__name__}({', '.join(items)})"