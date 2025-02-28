# install


on all nodes do this:
```bash
conda create -n base python=3.12.8 -y
conda activate base
pip install -r requirements_base.txt
```

# run

start ray cluster on each node

start with the head node
```bash
conda activate base
ray start --head --port 6379
```
```bash
ray start --head \
--resources='{"verification_slot":100}' \
--port=6379 \
--object-manager-port=8076 \
--temp-dir=/dev/shm/ray
```

worker node/s
```bash
conda activate base
ray start --address=head_node_ip:6379
```

### start the inference workers (both to get rollouts from the policy and to compute logprobs from said rollouts)
##### this should be done on each node you want to use for inference.


for example, for a node with 8 GPUs, and using 7 for generation and 1 for logprob, you would do the following:
```bash
for i in (seq 0 7)
    if test $i -lt 7
        echo "Launching generation worker on GPU $i..."
        python worker_dispatcher.py \
            --model_path /dev/shm/qwen7b-math-base \
            --mode generation \
            --tensor_parallel_size 1 \
            --max_num_seqs 128 \
            --write_failed_generation_samples \
            --global_num_verifiers 100 | tee generation_worker_$i.log &
    else
        # CUDA_VISIBLE_DEVICES="$i" python logprob_worker.py --model_path /dev/shm/phi-4 &
        echo "Launching logprob worker on GPU $i..."
        torchrun --nproc_per_node=1 --master_port=1234$i worker_dispatcher.py \
            --model_path /dev/shm/qwen7b-math-base \
            --mode logprob | tee logprob_worker_$i.log &
    end
end
```

In our test, we used two nodes, a total of 16 GPUs, 14 for generation and 2 for logprob. you must wait until all the workers are started before starting the training, which is shown by `worker <ID> registered` for each worker. Adjust the number of verifiers, each uses one CPU, make sure your cluster has the capacity.

### start the training on the nodes you want to use for training, we've trained with 8 GPUs on a single training node.

```bash
conda create grpo python=3.12.8 -y; conda activate grpo;pip install -r requirements_fsdp.txt
torchrun --nproc_per_node=8 --master_port=12345 trainer_core.py 2>&1 | tee train_qwen.log
```

the hyperparameters to be tuned are in `trainer_core.py`.

### Troubleshooting

- when a ray worker fails, the driver (the process that spawns such worker) shows unrelated errors. It's usually a module not found error in the child worker.
- when things fail, do ray stop everywhere and restart the process on all nodes. Ray becomes a bit unstable when restarting processes.
- It's important to create a separate conda environment for the training process or the worker environments will become corrupted. The python version should be the same as the base environment.
- `ray list actors | grep ALIVE` can be used to check if all the expected workers are running.
- make sure you can do enough http connections on your cluster: `ulimit -n 65535`
- Ray creates a lot of temporary files in the `/tmp` directory. You can clean them up with `rm -rf /tmp/ray`. Also, you need enough space, otherwise use `ray start --temp-dir=/dev/shm/ray` to use the shared memory as a temporary directory.

### Architecture Explanation

#### System Overview

We use Ray to create a distributed system composed of workers with different roles. The main compute-bound workers are the training workers and the two types of inference workers (generation and logprob).

#### Sample Data Structure

The unit data structure is the concept of a `sample`. It is a dictionary that should contain `input_token_ids` (the [sample dataset](math_simplerl_qwen_data_token_ids.jsonl) contains it). Thus, every worker processes one unit of data at a time but asynchronously.

#### Inference Worker Registration

Both inference workers register themselves with a registry (defined in [`vllm_registry.py`](vllm_registry.py)). The purpose of the vllm registries is to manage requests sent to the separate pools of workers. They relay requests and load balance across the workers by sending the requests to the worker handling the least amount of data at a time. There is one copy of this process for each type (generation and logprob), and it is created only by the first inference worker to request it.

We also use [worker_dispatcher.py](worker_dispatcher.py) to dispatch the workers. This was needed to isolate the python environment of the two types of inference workers.

#### Generation Worker

The generation worker ([`GenerationVLLMWorker`](vllm_worker.py)) is responsible for generating rollouts. It uses a vllm asynchronous engine to generate these rollouts, and then it utilizes HF's [math-verify](https://github.com/huggingface/Math-Verify) to compute a reward (it expects a `gt_answer` key in the sample dict). This worker also completes most of the sample dict, including defining the IDs used for reference logprobs and training (`sample_position_ids`), as well as the `advantages` used in GRPO (or the normalized rewards across a sample’s rollouts).

#### Logprob Worker

The logprob worker ([`LogprobWorker`](logprob_worker.py)) computes the log probabilities of the rollouts. It uses the same function as the training process to compute the log probabilities ([`PerTokenLogProbsFromCE`](grpo_loss.py)) and leverages the same utility functions to process the samples into input IDs, position IDs, and labels (e.g. [`get_input_for_logprobs`](sample_processing_utils.py)). It loads the model in the same way as the training process (via [`setup_model`](setup_model.py)) but does not wrap it with FSDP. It also accumulates samples in a batch [`_centralize_inference_requests`] until a maximum number of tokens per GPU is reached to keep the GPUs usage at max.

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

Our GRPO loss is mathematically equivalent to that described in the original [GRPO paper](https://arxiv.org/pdf/2402.03300) — specifically, equation 19 (further simplified by having $\pi_{\theta_{\text{old}}} = \pi_\theta$, as the policy isn’t updated more than once per batch, aligning with equation 20’s gradient).

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
