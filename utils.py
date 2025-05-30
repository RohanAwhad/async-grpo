import importlib
import os
import logging
import inspect
from datetime import timedelta
from typing import Any

from rich.logging import RichHandler
import torch
from torch.distributed import get_rank, is_initialized

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor # FSDP2 uses DTensor

# Globals for parameter norm/delta calculation
_prev_params_cpu_for_combined_calc = []
_is_first_combined_calculation_step = True

def init_distributed_environment(args):
    from vllm_experience_batcher import get_or_create_experience_batcher
    import ray
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    args.local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl", timeout=timedelta(minutes=180))
    args.global_rank = dist.get_rank()
    tensor = torch.ByteTensor([False]).cuda()
    dist.all_reduce(tensor)
    dist.barrier()
    
    ray.init(address="auto", namespace="test")
    args.batcher_actor = get_or_create_experience_batcher(args.experience_batcher_name)
    args.actor_registry = ray.get_actor("generation_vllm_registry")
    args.reference_registry = ray.get_actor("logprob_vllm_registry")
    ray.get(args.batcher_actor.register_training_process.remote(
        args.global_rank,
        args.max_tokens_per_gpu,
        args.train_minibatch_size
    ))
    dist.barrier()
    # Initialize FSDP norm/delta globals after distributed setup
    global _prev_params_cpu_for_combined_calc, _is_first_combined_calculation_step
    _prev_params_cpu_for_combined_calc = []
    _is_first_combined_calculation_step = True

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
    
@torch.no_grad()
def get_fsdp_param_l1_stats(
    model: torch.nn.Module,
    initialize_baseline: bool = False
) -> tuple[float, float, float, float, float, float]:
    """
    Computes the global L1 norm of current model parameters,
    the L1 norm of the change since the previous call,
    the mean per-parameter relative update ratio,
    and the 25th, 50th, and 75th percentiles of those ratios.
    Uses float64 accumulation for numeric precision.

    Returns:
        tuple[float, float, float, float, float, float]:
            current_L1_norm,
            delta_L1_norm,
            mean_update_ratio,
            update_ratio_p25,
            update_ratio_p50,
            update_ratio_p75
    """
    global _prev_params_cpu_for_combined_calc, _is_first_combined_calculation_step
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    current_device = torch.cuda.current_device()

    # Prepare double-precision accumulators for L1 norms
    local_sum_abs_current_norms = torch.tensor(0.0, device=current_device, dtype=torch.float64)
    local_sum_abs_delta_norms   = torch.tensor(0.0, device=current_device, dtype=torch.float64)

    # Prepare lists to collect each shard's norms (to reconstruct full-param norms)
    local_abs_shard_list   = []
    local_delta_shard_list = []

    # Collect per-parameter update ratios
    per_param_update_ratios = []

    # Will hold this step's CPU-cached parameters
    new_prev_params_this_step_cpu = []

    # Handle initialization logic
    if initialize_baseline:
        _prev_params_cpu_for_combined_calc = []
        _is_first_combined_calculation_step = False
    elif _is_first_combined_calculation_step:
        raise RuntimeError(
            "get_fsdp_param_l1_stats called without prior initialization."
        )
    #########################################################
    # Loop processes each parameter: computes current & delta L1 norms, accumulates stats, and caches CPU copies.
    #########################################################
    for idx, current_param_fsdp in enumerate(model.parameters()):
        # Retrieve previous param if available
        prev_param_cpu_data = (
            _prev_params_cpu_for_combined_calc[idx]
            if not initialize_baseline and idx < len(_prev_params_cpu_for_combined_calc)
            else None
        )

        # Extract tensor for norm computation and storage
        if isinstance(current_param_fsdp, DTensor):
            local_tensor = current_param_fsdp.to_local()
            data_tensor = local_tensor if local_tensor.is_floating_point() else None
            storage_cpu = local_tensor.clone().detach().cpu() if local_tensor.is_floating_point() else None
        elif isinstance(current_param_fsdp, torch.Tensor):
            data_tensor = (
                current_param_fsdp.data.float() if rank == 0 and current_param_fsdp.is_floating_point() else None
            )
            storage_cpu = (
                current_param_fsdp.data.clone().detach().cpu() if rank == 0 and current_param_fsdp.is_floating_point() else None
            )
        else:
            data_tensor = storage_cpu = None

        # Compute L1 norm of current parameters
        if data_tensor is not None:
            current_abs_norm = data_tensor.double().abs().sum()
            local_sum_abs_current_norms += current_abs_norm
            current_abs_norm_val = current_abs_norm.item()
        else:
            current_abs_norm = torch.tensor(0.0, dtype=torch.float64, device=current_device)
            current_abs_norm_val = 0.0

        # Compute L1 norm of delta
        if (
            not initialize_baseline
            and prev_param_cpu_data is not None
            and data_tensor is not None
        ):
            prev_tensor = prev_param_cpu_data.to(data_tensor.device, non_blocking=True).double()
            delta_abs_norm = (data_tensor.double() - prev_tensor).abs().sum()
            local_sum_abs_delta_norms += delta_abs_norm
            delta_abs_norm_val = delta_abs_norm.item()
        else:
            delta_abs_norm_val = 0.0

        new_prev_params_this_step_cpu.append(storage_cpu)
        # Collect this shard's norms
        local_abs_shard_list.append(current_abs_norm_val)
        local_delta_shard_list.append(delta_abs_norm_val)

    # Update the global store of previous parameters
    _prev_params_cpu_for_combined_calc = new_prev_params_this_step_cpu

    # Reduce across ranks
    dist.all_reduce(local_sum_abs_current_norms, op=dist.ReduceOp.SUM)
    if not initialize_baseline:
        dist.all_reduce(local_sum_abs_delta_norms, op=dist.ReduceOp.SUM)

    # Extract final scalar norms for the full model
    current_L1_norm = local_sum_abs_current_norms.item()
    delta_L1_norm   = local_sum_abs_delta_norms.item() if not initialize_baseline else 0.0

    # Reconstruct per-parameter full norms by summing shards, then compute ratios
    abs_tensor = torch.tensor(local_abs_shard_list, dtype=torch.float64, device=current_device)
    delta_tensor = torch.tensor(local_delta_shard_list, dtype=torch.float64, device=current_device)
    dist.all_reduce(abs_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(delta_tensor, op=dist.ReduceOp.SUM)
    ratio_tensor = delta_tensor / (abs_tensor + 1e-12)
    min_update_ratio, p25, p50, p75, max_update_ratio = torch.quantile(
        ratio_tensor,
        torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64, device=current_device),
    ).tolist()

    return (
        current_L1_norm,
        delta_L1_norm,
        min_update_ratio,
        p25,
        p50,
        p75,
        max_update_ratio,
    )