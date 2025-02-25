### Architecture Explanation

#### System Overview

We use Ray to create a distributed system composed of workers with different roles. The main compute-bound workers are the training workers and the two types of inference workers (generation and logprob).

#### Sample Data Structure

The unit data structure is the concept of a `sample`. It is a dictionary that should contain `input_token_ids` (the [sample dataset](math_simplerl_qwen_data_token_ids.jsonl) contains it). Thus, every worker processes one unit of data.

#### Inference Worker Registration

Both inference workers register themselves with a registry (defined in [`vllm_registry.py`](vllm_registry.py)). The purpose of the vllm registries is to manage requests sent to the separate pools of workers. They relay requests and load balance across the workers by sending the requests to the worker handling the least amount of data at a time. There is one copy of this process for each type (generation and logprob), and it is created only by the first inference worker to request it.

#### Generation Worker

The generation worker ([`GenerationVLLMWorker`](vllm_worker.py)) is responsible for generating rollouts. It uses a vllm asynchronous engine to generate these rollouts, and then it utilizes HF's [math-verify](https://github.com/huggingface/Math-Verify) to compute a reward (it expects a `gt_answer` key in the sample dict). This worker also completes most of the sample dict, including defining the IDs used for reference logprobs and training (`sample_position_ids`), as well as the `advantages` used in GRPO (or the normalized rewards across a sample’s rollouts).

#### Logprob Worker

The logprob worker ([`LogprobWorker`](logprob_worker.py)) computes the log probabilities of the rollouts. It uses the same function as the training process to compute the log probabilities ([`PerTokenLogProbsFromCE`](grpo_loss.py)) and leverages the same utility functions to process the samples into input IDs, position IDs, and labels (e.g. [`get_input_for_logprobs`](sample_processing_utils.py)). It loads the model in the same way as the training process (via [`setup_model`](setup_model.py)) but does not wrap it with FSDP. Note that as of Feb 24th, it processes log probabilities one sample at a time, making it a current bottleneck in the training process.

#### Updating Inference Worker Weights

Both the generation and logprob workers have a method to update their weights. The main training process invokes this method through [`update_vllm_worker_weights`](trainer_core.py) to ensure that both the generation models (actors) and the logprob models (reference) are updated accordingly.

#### Experience Batcher

The main process interfacing with the inference workers is the [`ExperienceBatcher`](vllm_experience_batcher.py). There is only one instance of this process, which is created by the training process. Its responsibilities include:

- **Accumulating Batches:** Gathering batches of samples from each training process and sending them for rollouts and logprob computation ([`get_experience_and_ref_logprobs`](vllm_experience_batcher.py)).
- **Dispatching Minibatches:** Receiving responses from inference workers and distributing minibatches to the training processes ([`_create_batches`](vllm_experience_batcher.py)).
- **Batch Optimization:** Creating batches to maximize the use of GPU token limits while ensuring each training process receives at least one sample ([`add_sample_to_batches`](vllm_experience_batcher.py)).
- **Minimizing Downtime:** Dispatching batches promptly as soon as inference responses are complete.

#### Training Process

The [train](trainer_core.py) script orchestrates the entire workflow:

- **Model Setup and Wrapping:** Sets up the model, wraps it in FSDP, and creates the training objects (e.g., optimizer, learning rate scheduler).
- **Data Loading:** Constructs an infinite sampler for the dataloader, ensuring each rank receives a distinct portion of the dataset.
- **Batch Processing:** Samples batches from the dataloader and sends all samples to the experience batcher. It then asynchronously waits for the batches to be returned with rollouts and logprobs.
- **Loss Computation:** Computes the [GRPO loss](grpo_loss.py), which calculates a per-token loss for each sample in the batch, sums them (without averaging), and scales the loss by the number of training processes (to compensate for FSDP's mean reduction).
- **Gradient Scaling:** Before performing a gradient step, scales the gradients ([`take_gradient_step`](trainer_core.py)) by the total number of samples in the batch across all GPUs, ensuring the mathematical equivalence of processing the entire batch in one forward/backward pass.
- **Model Update:** Executes the gradient step and saves the model to disk.

#### GRPO Loss Details

Our GRPO loss is mathematically equivalent to that described in the original [GRPO paper](https://arxiv.org/pdf/2402.03300) — specifically, equation 19 (further simplified by having \(\pi_{\theta_{\text{old}}} = \pi_\theta\), as the policy isn’t updated more than once per batch, aligning with equation 20’s gradient).

To compute this loss:

1. **Batch Postprocessing:**  
   We use [post_process_batch](sample_processing_utils.py) to pack all samples into a single-dimensional `input tensor` for flash-attention, padding-free training (see [this](https://huggingface.co/blog/packing-with-FA2)). This function also produces an `output_indices` tensor indicating output token positions and broadcasts constants (like each sample's advantage and output length) across the batch.

2. **Log Probability Computation:**  
   With the postprocessed batch, we compute per-token log probabilities by slightly modifying the typical cross-entropy loss function ([`PerTokenLogProbsFromCE`](grpo_loss.py)). This leverages that the log probability of a token is equivalent to the negative cross-entropy loss for that token. Labels are set to \(-100\) for all tokens except the output tokens, ensuring that non-output tokens contribute 0 to the loss.

3. **Policy Gradient Loss:**  
   We compute the per-token policy gradient loss using:  $L_{pg} = -A_i \log \pi_\theta(a_t|a_{1:t-1})$  
   This serves as a per-token Taylor expansion approximation of the KL divergence (approximately $\pi_{\text{ref}}/\pi_\theta - \log(\pi_{\text{ref}}/\pi_\theta) - 1$). The losses are divided by the number of tokens in the output (as the RL loss is computed at the trajectory level) and summed across all samples (without averaging).

4. **Scaling Adjustments:**  
   The loss is scaled up by the number of training processes to counteract FSDP’s mean reduction, and the gradients are scaled down by the total number of samples (across all training processes) to effectively average across the batch.
