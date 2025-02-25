from copy import deepcopy
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, get_scheduler
from accelerate import Accelerator
from torch.distributed.fsdp import BackwardPrefetch, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial

from utils import log_rank_0
from grpo_loss import PerTokenLogProbsFromCE

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

def get_fsdp_config(args, model: PreTrainedModel):
    # Third Party
    from accelerate.utils import FullyShardedDataParallelPlugin
    from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload

    block_name = model._no_split_modules[0]
    fsdp_plugin = FullyShardedDataParallelPlugin(
        auto_wrap_policy=partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                get_module_class_from_name(model, block_name),
            },
        ),
        limit_all_gathers=True,
        mixed_precision_policy=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        sharding_strategy=ShardingStrategy[args.fsdp_sharding_strategy],
        # sync_module_states=True,
        # param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False),
        #     # if torch.distributed.get_rank()!=0 else None,
        # cpu_ram_efficient_loading=True,
        use_orig_params=True,
        state_dict_type="full_state_dict",
    )

    return fsdp_plugin


def setup_accelerator(args, model: PreTrainedModel):
    accelerator = Accelerator(
        fsdp_plugin=get_fsdp_config(args, model),
    )
    accelerator.even_batches = False
    return accelerator

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
    model.loss_function = PerTokenLogProbsFromCE

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

    model.gradient_checkpointing_enable()
    # torch.compile(model)
    return model

def setup_training_components(args, model):
    accelerator = setup_accelerator(args, model)
    model = accelerator.prepare(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
    )
    # Necessary so that Accelerate does not step once per GPU
    # see https://github.com/huggingface/accelerate/blob/127818fc27ebe5cb236357fff59ff1748326d643/src/accelerate/scheduler.py#L69
    lr_scheduler.split_batches = True
    accelerator.register_for_checkpointing(lr_scheduler)
    return model, accelerator, optimizer, lr_scheduler