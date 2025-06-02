from copy import deepcopy
import math
import torch
import torch.distributed
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, get_scheduler
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from utils import log_rank_0
from grpo_loss import make_grpo_forward, ce_loss_and_entropy_logsoftmax, grpo_loss_and_entropy_ce_from_logsoftmax

def get_module_class_from_name(
    model: torch.nn.Module, name: str
) -> torch.nn.Module | None:
    modules_children = list(model.children())

    if model.__class__.__name__ == name:
        return model.__class__
    elif len(modules_children) == 0:
        return
    else:
        for child_module in modules_children:
            module_class = get_module_class_from_name(child_module, name)
            if module_class is not None:
                return module_class

def align_model_and_tokenizer(model, tokenizer):
    """
    Aligns the model's vocabulary and special tokens with the tokenizer.
    """
    if len(tokenizer) > model.config.vocab_size:
        print(
            f"WARNING: tokenizer has {len(tokenizer)} tokens but model has {model.config.vocab_size} vocab size"
        )
        model.resize_token_embeddings(
            int(8 * math.ceil(len(tokenizer) / 8.0))
        )  # make the vocab size multiple of 8 for sharding the embedding layer.

    # Fix any discrepancy between model and tokenizer
    special_tokens = {
        'pad': ('pad_token_id', 'Fixing model pad token id'),
        'bos': ('bos_token_id', 'Fixing model bos token id'),
        'eos': ('eos_token_id', 'Fixing model eos token id')
    }

    for token_type, (token_attr, message) in special_tokens.items():
        model_token = getattr(model.config, token_attr)
        tokenizer_token = getattr(tokenizer, token_attr)
        
        if (model_token is not None and tokenizer_token is not None 
            and model_token != tokenizer_token):
            log_rank_0(
                "\033[38;5;226m"
                f"WARNING: There is a mismatch between {token_type} token id of "
                f"model({model_token}) and tokenizer({tokenizer_token}). "
                f"{message} to be same as tokenizer's {token_type} token id"
                "\033[0m"
            )
            setattr(model.config, token_attr, tokenizer_token)

    return model


def setup_model(args, model=None):

    base_model_args = {
        "pretrained_model_name_or_path": args.model_name_or_path,
        "torch_dtype": torch.bfloat16,
    }
    base_model_args["attn_implementation"] = "flash_attention_2"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(**base_model_args)
    model = align_model_and_tokenizer(model, tokenizer)
    model.gradient_checkpointing_enable()
    
    mode = getattr(args, 'mode', 'eval')
    temperature = getattr(args, 'temperature', 1.0)
    model = make_grpo_forward(model, temperature, mode=mode, use_torch_compile=getattr(args, 'use_torch_compile', False))

    if model.__class__.__name__ not in [
        "MistralForCausalLM",
        "GPTDolomiteForCausalLM", 
        "LlamaForCausalLM",
        "Starcoder2ForCausalLM",
        "GemmaForCausalLM",
        "MixtralForCausalLM",
        "GraniteForCausalLM",
    ]:
        log_rank_0(
            f"\033[38;2;255;255;0mWarning: Model class name: {model.__class__.__name__} is not in the list of supported models.\033[0m",
            to_print=True,
        )

    # torch.compile(model)
    return model

def wrap_fsdp2(model: torch.nn.Module, use_torch_compile: bool = False) -> torch.nn.Module:
    """
    Wrap `model` in PyTorch FSDP2 with full sharding and transformer auto-wrap policy under BF16.
    """
    # Determine the block class to auto-wrap (first no-split module)
    block_name = model._no_split_modules[0]
    block_cls = get_module_class_from_name(model, block_name)
    if block_cls is None:
        raise ValueError(f"Could not find module class named {block_name}")
    # Mixed-precision policy for BF16
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
        # reduce_dtype=torch.bfloat16,
        # output_dtype=torch.bfloat16,
        cast_forward_inputs=True,
    )
    # FSDP2 settings: full shard, BF16, no CPU offload
    fsdp2_kwargs = {
        "mp_policy": mp_policy,
        "reshard_after_forward": True,
    }
    # Auto-wrap child modules
    for module in model.modules():
        if isinstance(module, block_cls):
            fully_shard(module, **fsdp2_kwargs)
    # Wrap the full model
    fully_shard(model, **fsdp2_kwargs)
    # Cast back to float32
    model = model.to(torch.float32)
    if use_torch_compile:
        model = torch.compile(model)
    return model

def setup_training_components(args, model):
    model = wrap_fsdp2(model, args.use_torch_compile)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
    )
    lr_scheduler.split_batches = True
    lr_scheduler.step()  # the scheduler starts at 0 and there's no learning.
    return model, optimizer, lr_scheduler