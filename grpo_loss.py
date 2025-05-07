import os
import time
import torch
torch.set_float32_matmul_precision('high')
from typing import Optional, Tuple, Union, List
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast

DEBUG = False
def debug_print(message):
    if DEBUG:
        rank = torch.distributed.get_rank()
        with open(f"debug_grpo_loss_{rank}.log", "a") as f:
            f.write(message + "\n")
    print(message)


def make_grpo_forward(model, loss_chunksize: int = None):
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
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        nonlocal loss_chunksize
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

        T = hidden_states.shape[1]

        if loss_chunksize is None or labels is None:
            loss_chunksize = 2**31 - 1  # max value for Python's int
        
        loss_chunksize = min(loss_chunksize, T)

        # shift labels by one to the left so input_ids[0] predicts labels[1] and pad with -100 on the right since the last input_id doesn't correspond to any label.
        shifted_labels = nn.functional.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
        total_loss = []
        entropy_list = []

        for i in range(0, T, loss_chunksize):
            end_idx = min(i + loss_chunksize, T)
            logits = model.lm_head(hidden_states[:, i:end_idx, :]).float()
            loss, logits = model.loss_function(logits=logits, labels=shifted_labels[:, i:end_idx], vocab_size=model.config.vocab_size, **kwargs)
            total_loss.append(loss)
            entropy_list.append(entropy_from_logits(logits.detach().bfloat16()))
        total_loss = torch.cat(total_loss, dim=0)
        debug_print(f"\033[1;38;2;255;165;0m _forward line 79: \033[0m max total_loss: {max(torch.abs(total_loss.detach()))}")
        debug_print(f"\033[1;38;2;255;165;0m _forward line 79: \033[0m mean total_loss: {total_loss.detach().mean()}")

        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=torch.cat(entropy_list, dim=0), #using the logits field to store the entropy for metrics.
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    model.__original_forward = model.forward
    model.forward = _forward
    return model

# see transformers/loss/loss_utils.py:ForCausalLMLoss
# coming from the fact that logprobs equivalent to -CrossEntropyLoss(logits, labels)
# this is a modification that does exactly the same except that there's no reduction
# and we return the per-token log probabilities as -CrossEntropyLoss(logits, labels)
def PerTokenLogProbsFromCE(
    logits, labels, vocab_size: int, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs
):
    """
    Compute per-token log probabilities from a cross-entropy loss.
    returns a tensor of shape (L,) where L is the total number of tokens in the batch.
    the logprob i corresponds to the token at position i+1 in the input_ids.
    the logprob at position -1 is 0, as well as the logprob at position len(sample)-1 of each sample.
    """
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    labels = labels.to(logits.device)
    # Shift so that tokens < n predict n
    # labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
    # shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    
    labels = labels.view(-1)
    # Enable model parallelism
    labels = labels.to(logits.device)
    per_token_ce = nn.functional.cross_entropy(logits, labels, reduction="none", ignore_index=ignore_index)
    logprobs = -per_token_ce
    return logprobs, logits

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

def compute_kl_divergence(
    policy_logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor
) -> torch.Tensor:
    """
    Compute KL divergence between policy and reference model using the Schulman approximation.
    KL ≈ π_ref/π_θ - log(π_ref/π_θ) - 1
    """
    ratio = torch.exp(reference_logprobs - policy_logprobs)
    kl_div = ratio - (reference_logprobs - policy_logprobs) - 1
    return kl_div

# def get_mean_per_sample_loss(loss, output_lens_broadcasted, num_samples):
#     """
#     loss is a tensor of shape [1, N] where N is the total number of tokens across all samples.
#     output_lens_broadcasted has the length
#     """
#     return (loss/output_lens_broadcasted).sum()/num_samples

@torch.compile
def entropy_from_logits(logits: torch.Tensor):
    """Calculate entropy from logits."""
    with torch.no_grad():
        pd = torch.nn.functional.softmax(logits, dim=-1)
        entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy

# @torch.compile
def compute_grpo_loss(
    policy_model,
    minibatch,
    kl_coeff: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute GRPO loss with its components using the PPO-style probability ratio trick.

    # The policy gradient is computed as:
    #
    #     ∇θ J(πθ) = 𝔼₍τ ∼ πθ₎ [ Σₜ₌₀ᵀ ∇θ log πθ(aₜ | sₜ) · Φₜ ]
    #
    # Here, ∇θ denotes the gradient with respect to the policy parameters θ, and the expectation
    # is over trajectories τ sampled from the policy πθ. In our implementation, we then take an
    # average over the number of trajectories in the batch (at the gradient step level).
    # We also divide by the number of tokens (actions) in each trajectory to ensure long and short
    # trajectories contribute to the gradient equally.
    """
    # torch.autograd.set_detect_anomaly(True)
    # print("\033[1;91;40mDEBUG: remove torch.autograd.set_detect_anomaly(True)\033[0m")
    batch_ids = minibatch["batch_ids"]
    batch_position_ids = minibatch["batch_position_ids"]
    output_indices = minibatch["output_indices"]
    reference_logprobs = minibatch["reference_output_logprobs"]
    advantages = minibatch["advantages"]
    output_lens_broadcasted = minibatch["output_lens_broadcasted"]
    labels = minibatch["labels"]


    model_out = policy_model(
        input_ids=batch_ids,
        position_ids=batch_position_ids,
        labels=labels,
        use_cache=False,
    )

    policy_logprobs = model_out.loss
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
    
    # this is equal to 1 but it keeps the gradient flowing through the policy_logprobs without affecting the value of the pg_loss (it's only the advantages)
    prob_ratio = torch.exp(policy_logprobs - policy_logprobs.detach()) # this is 1
    pg_loss = -(prob_ratio * advantages) # equal to -advantages since we are maximizing the advantages.
    
    # KL penalty term using the improved approximation
    kl_div = compute_kl_divergence(policy_logprobs, reference_logprobs)
    num_clamped = (torch.abs(kl_div) > 10).sum().item()
    kl_div = torch.clamp(kl_div, min=-10, max=10)
    debug_print(f"\033[1;38;2;255;165;0m _compute_grpo_loss line 262: \033[0m num_clamped: {num_clamped}")
    
    # Combined loss
    loss = pg_loss + kl_coeff * kl_div
    
    loss_metrics = (loss.detach()).sum().item()
    pg_loss_metrics = (pg_loss.detach()).sum().item()
    kl_div_metrics = (kl_div.detach()).sum().item()
    start_time = time.time()
    entropy_metrics = model_out.logits[output_indices].sum()
    # print(f"Time taken to compute entropy: {time.time() - start_time} seconds", flush=True)
    loss = loss.sum()
    # torch.distributed.breakpoint()

    return loss, loss_metrics, pg_loss_metrics, kl_div_metrics, entropy_metrics
