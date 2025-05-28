from dataclasses import dataclass
from functools import partial
import os
import time
import torch
torch.set_float32_matmul_precision('high')
from typing import Callable, Optional, Tuple, Union, List
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput
import torch.nn.functional as F

DEBUG = False
def debug_print(message):
    if DEBUG:
        rank = torch.distributed.get_rank()
        with open(f"debug_grpo_loss_{rank}.log", "a") as f:
            f.write(message + "\n")
    print(message)

@dataclass
class GRPOOutput(ModelOutput):
    loss: torch.Tensor = None
    entropy: torch.Tensor = None
    neg_log_ratio: torch.Tensor = None
    loss_clip1: torch.Tensor = None
    loss_unclipped: torch.Tensor = None
    loss_clipped: torch.Tensor = None

@dataclass
class LogProbsOutput(ModelOutput):
    loss: torch.Tensor = None
    entropy: torch.Tensor = None


def make_grpo_forward(model, temperature: float = 1.0, mode: str = 'training', use_torch_compile: bool = True):
    def _forward(
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        old_logprobs: Optional[torch.Tensor] = None,
        advantages: Optional[torch.Tensor] = None,
        clip_low: float = 0.2,
        clip_high: float = 0.4,
        clip_ratio_c: float = 1e32,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else model.config.use_return_dict
        debug_print(f"\033[1;38;2;255;165;0m _forward line 36: \033[0m input_ids.shape: {input_ids.shape}")
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs[0]

        debug_print(f"\033[1;38;2;255;165;0m _forward line 53: \033[0m hidden_states.shape: {hidden_states.shape}")

        # T = hidden_states.shape[1]

        # if loss_chunksize is None or labels is None:
        #     loss_chunksize = 2**31 - 1  # max value for Python's int
        
        # loss_chunksize = min(loss_chunksize, T)

        # shift labels by one to the left so input_ids[0] predicts labels[1] and pad with -100 on the right since the last input_id doesn't correspond to any label.
        # shifted_labels = nn.functional.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
        # total_loss = []
        # entropy_list = []

        # for i in range(0, T, loss_chunksize):
        #     end_idx = min(i + loss_chunksize, T)
        #     logits = model.lm_head(hidden_states[:, i:end_idx, :]).float().div_(temperature)
        #     loss, logits = model.loss_function(logits=logits, labels=shifted_labels[:, i:end_idx], vocab_size=model.config.vocab_size, **kwargs)
        #     total_loss.append(loss)
        #     entropy_list.append(entropy_from_logits(logits.detach().bfloat16()))
        # total_loss = torch.cat(total_loss, dim=0)
        # debug_print(f"\033[1;38;2;255;165;0m _forward line 79: \033[0m max total_loss: {max(torch.abs(total_loss.detach()))}")
        # debug_print(f"\033[1;38;2;255;165;0m _forward line 79: \033[0m mean total_loss: {total_loss.detach().mean()}")

        if mode == 'training':
            # select the GRPO loss function, compile if requested, without shadowing the global
            loss_fn = grpo_loss_and_entropy_ce_from_logsoftmax
            if use_torch_compile:
                loss_fn = torch.compile(loss_fn)
            
            out = loss_fn(
                lm_head_weights=model.lm_head.weight,
                lm_head_bias=model.lm_head.bias if hasattr(model.lm_head, "bias") else None,
                hidden_states=hidden_states,
                labels=labels,
                temperature=temperature,
                old_logprobs=old_logprobs,
                advantages=advantages,
                clip_low=clip_low,
                clip_high=clip_high,
                clip_ratio_c=clip_ratio_c,
            )
            return GRPOOutput(
                loss=out[0],
                entropy=out[1],
                neg_log_ratio=out[2],
                loss_clip1=out[3],
                loss_unclipped=out[4],
                loss_clipped=out[5],
            )
        elif mode == 'eval':
            # select the eval log-prob function, compile if requested
            loss_fn = ce_loss_and_entropy_logsoftmax
            if use_torch_compile:
                loss_fn = torch.compile(loss_fn)
            
            out = loss_fn(
                lm_head_weights=model.lm_head.weight,
                lm_head_bias=model.lm_head.bias if hasattr(model.lm_head, "bias") else None,
                hidden_states=hidden_states,
                labels=labels,
                temperature=temperature,
            )
            return LogProbsOutput(
                loss=out[0],
                entropy=out[1],
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
    # if use_torch_compile:
    #     model.model = torch.compile(model.model)
    
    model.__original_forward = model.forward
    model.forward = _forward
    return model

# see transformers/loss/loss_utils.py:ForCausalLMLoss
# coming from the fact that logprobs equivalent to -CrossEntropyLoss(logits, labels)
# this is a modification that does exactly the same except that there's no reduction
# and we return the per-token log probabilities as -CrossEntropyLoss(logits, labels)
# @torch.compile
# def PerTokenLogProbsFromCE(
#     logits, labels, vocab_size: int, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs
# ):
#     """
#     Compute per-token log probabilities from a cross-entropy loss.
#     returns a tensor of shape (L,) where L is the total number of tokens in the batch.
#     the logprob i corresponds to the token at position i+1 in the input_ids.
#     the logprob at position -1 is 0, as well as the logprob at position len(sample)-1 of each sample.
#     """
#     # Upcast to float if we need to compute the loss to avoid potential precision issues
#     logits = logits.float()
#     labels = labels.to(logits.device)
#     # Shift so that tokens < n predict n
#     # labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
#     # shift_labels = labels[..., 1:].contiguous()

#     # Flatten the tokens
#     logits = logits.view(-1, vocab_size)
    
#     labels = labels.view(-1)
#     # Enable model parallelism
#     labels = labels.to(logits.device)
#     per_token_ce = nn.functional.cross_entropy(logits, labels, reduction="none", ignore_index=ignore_index)
#     logprobs = -per_token_ce
#     return logprobs, logits

# def get_per_token_logps(model, batched_samples_ids, batched_samples_position_ids, batched_samples_output_indices):
#     """
#     Given logits and target token IDs, compute per-token log probabilities
#     using torch.nn.functional.cross_entropy with reduction='none'.

#     Args:
#         logits (torch.Tensor): Tensor of shape [B, L, V] (B=batch size, L=sequence length, V=vocab size).
#         target (torch.Tensor): Tensor of shape [B, L] containing target token IDs.
#         ignore_index (int): The index that should be ignored in the loss computation.

#     Returns:
#         torch.Tensor: Per-token log probabilities of shape [B, L].
#     """
#     # Assume tokens to predict are shifted by one:
#     #   logits: prediction for t+1 tokens -> remove last time step
#     shift_logits = logits[:, :-1, :]             # shape: [B, L-1, V]
#     shift_target = target[:, 1:].contiguous()      # shape: [B, L-1]

#     # Flatten for cross entropy computation.
#     flat_logits = shift_logits.reshape(-1, shift_logits.size(-1))
#     flat_target = shift_target.reshape(-1)

#     # Compute element-wise cross entropy loss (-log probability).
#     losses = F.cross_entropy(
#         flat_logits, flat_target, reduction="none", ignore_index=ignore_index
#     )  # shape: [B*(L-1)]

#     # Reshape back to [B, L-1]
#     losses = losses.view(shift_target.size())

#     # The log probabilities are the negatives of the loss values.
#     logprobs = -losses
#     return logprobs

#     # ALDO: batched_samples_output_indices index the logits
#     # but the logits correspond to the next token probabilities (logit i is the logit for token i+1)
#     # therefore we need to index the output_ids of the indices + 1
#     output_ids = batched_samples_ids[:, batched_samples_output_indices+1].contiguous().to(logits.device)
#     token_logits = logits.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1).contiguous()
#     # Compute logsumexp for normalization over the vocab dimension: shape [1, N]
#     # do a for loop to reduce memory peak
#     logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits]).contiguous()
#     # Final log probability per token: shape [1, N]
#     output_log_probs = token_logits - logsumexp_values
#     return output_log_probs

# def compute_kl_divergence(
#     policy_logprobs: torch.Tensor,
#     reference_logprobs: torch.Tensor
# ) -> torch.Tensor:
#     """
#     Compute KL divergence between policy and reference model using the Schulman approximation.
#     KL ≈ π_ref/π_θ - log(π_ref/π_θ) - 1
#     """
#     ratio = torch.exp(reference_logprobs - policy_logprobs)
#     kl_div = ratio - (reference_logprobs - policy_logprobs) - 1
#     return kl_div

# # def get_mean_per_sample_loss(loss, output_lens_broadcasted, num_samples):
# #     """
# #     loss is a tensor of shape [1, N] where N is the total number of tokens across all samples.
# #     output_lens_broadcasted has the length
# #     """
# #     return (loss/output_lens_broadcasted).sum()/num_samples

# # @torch.compile
# def entropy_from_logits(logits: torch.Tensor):
#     """Calculate entropy from logits."""
#     with torch.no_grad():
#         pd = torch.nn.functional.softmax(logits, dim=-1)
#         entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
#     return entropy

# # @torch.compile
# def policy_loss(
#     policy_logprobs: torch.Tensor,
#     old_logprobs: torch.Tensor,
#     advantages: torch.Tensor,
#     output_indices: torch.Tensor,
#     clip_low: float,
#     clip_high: float,
#     clip_ratio_c: float,
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#     negative_approx_kl = policy_logprobs - old_logprobs
#     ratio = torch.exp(negative_approx_kl)
#     # ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

#     pg_losses1 = -advantages * ratio
#     pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_low, 1 + clip_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
#     clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
#     # pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

#     pg_losses3 = -advantages * clip_ratio_c
#     clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
#     # pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

#     pg_loss = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1).sum()
#     # pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

#     # return pg_loss, negative_approx_kl, clip_pg_losses1, pg_losses1, 

#     ##########
#     neg_log_ratio = policy_logprobs - old_logprobs
#     ratio = torch.exp(neg_log_ratio)

#     # 2) Determine clipping bounds
#     lower = 1.0 - clip_low
#     upper = 1.0 + clip_high

#     # 3) Compute per-token surrogate losses
#     loss_unclipped = -advantages * ratio
#     loss_clipped = -advantages * torch.clamp(ratio, lower, upper)
#     loss_clipped_dual = -advantages * clip_ratio_c

#     # 4) Standard PPO clipping: max(unclipped, clipped)
#     loss_clip1 = torch.maximum(loss_unclipped, loss_clipped)

#     # 5) Dual-clip step for negative advantages: cap at clip_ratio_c
#     is_negative = advantages < 0
#     loss_clip2 = torch.minimum(loss_clip1, loss_clipped_dual)

#     # Final per-token loss selection
#     pg_loss = torch.where(is_negative, loss_clip2, loss_clip1).sum()
#     return pg_loss, neg_log_ratio, loss_clip1, loss_unclipped, loss_clipped, loss_clipped_dual, is_negative

def per_token_log_probs_from_log_softmax(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    ignore_index: int = -100,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-token log probabilities (no reduction) via log_softmax + gather.
    Returns (logprobs, logits_flat) where logprobs: [B*Lc] and logits_flat: [B*Lc, V].
    """
    # Upcast to float and move labels to correct device
    logits = logits.float()
    labels = labels.to(logits.device)

    # Flatten tokens
    logits_flat = logits.view(-1, vocab_size)
    labels_flat = labels.contiguous().view(-1)

    # Safe label indices for ignore positions
    safe_labels = labels_flat.clone()
    safe_labels[safe_labels == ignore_index] = 0

    # Compute log-probabilities
    logp_flat = F.log_softmax(logits_flat, dim=-1)

    # Gather log-probabilities of the true labels
    logprobs = logp_flat.gather(1, safe_labels.unsqueeze(1)).squeeze(1)

    # Zero out ignored positions
    logprobs = logprobs * (labels_flat != ignore_index)
    return logprobs, logits_flat

def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Calculate entropy for each row of logits: H(p) = logsumexp - sum(p*logits)."""
    with torch.no_grad():
        pd = F.softmax(logits, dim=-1)
        entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy


def fused_log_probs_and_entropy_fn(
    lm_head_weights: torch.Tensor,
    lm_head_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    per_token_fn: Callable = per_token_log_probs_from_log_softmax,
    **unused_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generic GRPO loss/entropy that uses a per-token logprobs function.
    """
    assert labels.shape[0] == 1, "ce_loss_and_entropy only supports batch size 1 but packed"
    shifted_labels = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
    V = lm_head_weights.size(0)
    logits = torch.matmul(hidden_states, lm_head_weights.t())
    if lm_head_bias is not None:
        logits = logits + lm_head_bias
    logits = logits.float().div_(temperature)

    loss_flat, logits_flat = per_token_fn(logits, shifted_labels, vocab_size=V)
    ent = entropy_from_logits(logits_flat.detach().bfloat16())
    return loss_flat, ent

ce_loss_and_entropy_logsoftmax = partial(
    fused_log_probs_and_entropy_fn, per_token_fn=per_token_log_probs_from_log_softmax
)

def grpo_loss_and_entropy_ce_fn(
    lm_head_weights: torch.Tensor,
    lm_head_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    old_logprobs: Optional[torch.Tensor] = None,
    advantages: Optional[torch.Tensor] = None,
    clip_low: float = 0.2,
    clip_high: float = 0.4,
    clip_ratio_c: float = 1e32,
    ce_loss_and_entropy_fn: Callable = ce_loss_and_entropy_logsoftmax,
    **unused_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-token negative log probabilities (GRPO loss) and entropies in one shot without chunking.
    """
    policy_logprobs, entropy = ce_loss_and_entropy_fn(
        lm_head_weights, lm_head_bias, hidden_states, labels, temperature
    )
    if old_logprobs is None:
        old_logprobs = policy_logprobs.detach()
    neg_log_ratio = policy_logprobs - old_logprobs
    ratio = torch.exp(neg_log_ratio)

    # 2) Determine clipping bounds
    lower = 1.0 - clip_low
    upper = 1.0 + clip_high

    # 3) Compute per-token surrogate losses
    loss_unclipped = -advantages * ratio
    loss_clipped = -advantages * torch.clamp(ratio, lower, upper)
    # loss_clipped_dual = -advantages * clip_ratio_c

    # 4) Standard PPO clipping: max(unclipped, clipped)
    loss_clip1 = torch.maximum(loss_unclipped, loss_clipped)

    # 5) Dual-clip step for negative advantages: cap at clip_ratio_c
    # is_negative = advantages < 0
    # loss_clip2 = torch.minimum(loss_clip1, loss_clipped_dual)

    # Final per-token loss selection
    # pg_loss = torch.where(is_negative, loss_clip2, loss_clip1)
    pg_loss = loss_clip1

    # mask out positions where shifted label is ignore_index (-100)
    shifted_labels = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
    mask = shifted_labels.view(-1) != -100
    pg_loss = torch.where(mask, pg_loss, torch.zeros_like(pg_loss))
    pg_loss = pg_loss.sum()
    return pg_loss, entropy, neg_log_ratio.detach(), loss_clip1.detach(), loss_unclipped.detach(), loss_clipped.detach()

grpo_loss_and_entropy_ce_from_logsoftmax = partial(
    grpo_loss_and_entropy_ce_fn, ce_loss_and_entropy_fn=ce_loss_and_entropy_logsoftmax
)

# @torch.compile
def compute_dual_clip_grpo_loss(
    policy_model,
    minibatch,
    clip_low: float,
    clip_high: float,
    clip_ratio_c: float,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute GRPO loss using the dual-clip PPO algorithm.

    This loss applies two clipping mechanisms:
      - Clipping the probability ratio within [1 - clip_low, 1 + clip_high].
      - For negative advantages, clipping the loss to -advantages * clip_ratio_c.

    Args:
        policy_model: A causal LM policy model to evaluate.
        minibatch: A dict containing batch data keys:
            'batch_ids', 'batch_position_ids', 'labels',
            'reference_output_logprobs', 'advantages', and 'output_indices'.
        clip_low: Lower epsilon for PPO clipping (bounds ratio >= 1 - clip_low).
        clip_high: Upper epsilon for PPO clipping (bounds ratio <= 1 + clip_high).
        clip_ratio_c: Dual clipping constant c for negative advantages.

    Returns:
        loss (Tensor): Scalar policy gradient loss to backpropagate.
        metrics (dict): Dictionary of metric values:
            'loss', 'pg_loss', 'pg_clip', 'pg_clip_lower', 'kl_div', 'entropy'.
    """
    # torch.autograd.set_detect_anomaly(True)
    # print("\033[1;91;40mDEBUG: remove torch.autograd.set_detect_anomaly(True)\033[0m")
    batch_ids = minibatch["batch_ids"]
    batch_position_ids = minibatch["batch_position_ids"]
    output_indices = minibatch["output_indices"]
    old_logprobs = minibatch["reference_output_logprobs"]
    advantages = minibatch["advantages"]
    output_lens_broadcasted = minibatch["output_lens_broadcasted"]
    labels = minibatch["labels"]



    grpo_output = policy_model.forward(
        input_ids=batch_ids,
        position_ids=batch_position_ids,
        labels=labels,
        use_cache=False,
        old_logprobs=old_logprobs,
        advantages=advantages,
        clip_low=clip_low,
        clip_high=clip_high,
        clip_ratio_c=clip_ratio_c,
    )


    # model_out = policy_model(
    #     input_ids=batch_ids,
    #     position_ids=batch_position_ids,
    #     labels=labels,
    #     use_cache=False,
    # )

    # policy_logprobs = model_out.loss
    ##### DEBUG #####
    # minibatch['samples'][0].keys()
    # minibatch['samples'][1]['sample_text']
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained(policy_model.config._name_or_path)
    # tokenizer.decode(batch_ids[:,output_indices+1][:,:100].squeeze().tolist())
    # tokenizer.decode(labels[labels!=-100][:100])
    # tokenizer.decode(batch_ids[labels!=-100][:100].squeeze().tolist())
    # tokenizer.decode(batch_ids[:, (reference_logprobs!=0) * (policy_logprobs==0)][:100].squeeze().tolist())
    # (batch_ids[labels!=-100] - labels[labels!=-100]).sum()
    # print(tokenizer.decode(batch_ids[0,:100].squeeze().tolist()))
    # print(tokenizer.decode(batch_ids[:,-1000:].squeeze().tolist()))
    # non_zero_logprobs = policy_logprobs[policy_logprobs != 0]
    # torch.abs(non_zero_logprobs).mean()
    # pol_ref = torch.abs(policy_logprobs - reference_logprobs)
    # kl_div[kl_div != 0].mean()
    # torch.distributed.breakpoint()
    # print(tokenizer.decode(batch_ids[:,115:120].squeeze().tolist()))
    # print(policy_logprobs[1164:1500])
    # print(reference_logprobs[1164:1500])
    # policy_logprobs[policy_logprobs != 0].shape
    # reference_logprobs[reference_logprobs != 0].shape
    # diff = torch.abs(policy_logprobs - reference_logprobs)/torch.abs((policy_logprobs+reference_logprobs)/2+1e-10)
    # idx = diff.argsort(descending=True)
    # diff[idx[:10]]
    # i = idx[0]
    # [print(f"reflogprobs: {reference_logprobs[i]}, policylogprobs: {policy_logprobs[i]}, diff: {diff[i]}") for i in idx[:10]]
    # debug_print(f"\033[1;38;2;255;165;0m _compute_grpo_loss line 272: \033[0m diff[diff!=0].mean(): {diff[labels[0]!=-100].mean()}")
    # print(tokenizer.decode(batch_ids[:,i-10:i+10].squeeze().tolist()))
    # diff[diff>1e-1].shape
    # (diff>1e-1).nonzero()
    # torch.distributed.breakpoint()

    ##### DEBUG #####
    # pg_loss, neg_log_ratio, loss_clip1, loss_unclipped, loss_clipped, loss_clipped_dual, is_negative = policy_loss(
    #     policy_logprobs=policy_logprobs,
    #     old_logprobs=old_logprobs,
    #     advantages=advantages,
    #     output_indices=output_indices,
    #     clip_low=clip_low,
    #     clip_high=clip_high,
    #     clip_ratio_c=clip_ratio_c,
    # )

    # Track clipping and loss metrics
    metrics = {
        "loss": grpo_output.loss.detach().item(),
        "pg_clip": (grpo_output.loss_clipped > grpo_output.loss_unclipped).detach()[output_indices].sum().item(),
        "kl_div": (-grpo_output.neg_log_ratio.detach())[output_indices].sum().item(),
        "entropy": grpo_output.entropy[output_indices].sum().item(),
    }
    # metrics = {
    #     "loss": pg_loss.detach().item(),
    #     "pg_loss": pg_loss.detach().item(),
    #     "pg_clip": (loss_clipped > loss_unclipped).detach().sum().item(),
    #     "pg_clip_lower": ((loss_clip1 > loss_clipped_dual) & is_negative).detach().sum().item(),
    #     "kl_div": (-neg_log_ratio.detach()).sum().item(),
    #     "entropy": model_out.logits[output_indices].sum().item(),
    # }
    return grpo_output.loss, metrics
    
    
    # # this is equal to 1 but it keeps the gradient flowing through the policy_logprobs without affecting the value of the pg_loss (it's only the advantages)
    # prob_ratio = torch.exp(policy_logprobs - policy_logprobs.detach()) # this is 1
    # pg_loss = -(prob_ratio * advantages) # equal to -advantages since we are maximizing the advantages.
    
    # # KL penalty term using the improved approximation
    # kl_div = compute_kl_divergence(policy_logprobs, reference_logprobs)
    # num_clamped = (torch.abs(kl_div) > 10).sum().item()
    # kl_div = torch.clamp(kl_div, min=-10, max=10)
    # debug_print(f"\033[1;38;2;255;165;0m _compute_grpo_loss line 262: \033[0m num_clamped: {num_clamped}")
    
    # # Combined loss
    # loss = pg_loss + kl_coeff * kl_div
    
    # loss_metrics = (loss.detach()).sum().item()
    # pg_loss_metrics = (pg_loss.detach()).sum().item()
    # kl_div_metrics = (kl_div.detach()).sum().item()
    # start_time = time.time()
    # entropy_metrics = model_out.logits[output_indices].sum()
    # # print(f"Time taken to compute entropy: {time.time() - start_time} seconds", flush=True)
    # loss = loss.sum()
    # # torch.distributed.breakpoint()

    # return loss, loss_metrics, pg_loss_metrics, kl_div_metrics, entropy_metrics
