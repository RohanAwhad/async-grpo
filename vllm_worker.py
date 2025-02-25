import argparse
from copy import deepcopy
from functools import partial
from hashlib import sha256
import random
import logging
import time
import ray
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.inputs import TokensPrompt
from transformers import AutoTokenizer
import torch
import uuid
import atexit
from vllm_registry import get_or_create_registry  # helper function from registry
import asyncio
from math_verify import parse, verify
from math_verify.parser import LatexExtractionConfig, NormalizationConfig
import re

import numpy as np
from numba import njit
logging.getLogger('numba').setLevel(logging.WARNING)

@njit
def normalize_rewards(rewards):
    """
    Normalize rewards within each group to compute advantages.

    Parameters:
        rewards : np.ndarray (1D)
            Array of rewards for each sample of shape (n_samples,).

    Returns:
        np.ndarray (1D)
            Normalized rewards of shape (n_samples,).
    """
    mean = np.mean(rewards)
    std = np.std(rewards) + 1e-4
    return (rewards - mean) / std



class BaseVLLMWorker:
    """
    Base vLLM worker class encapsulating common logic.
    Inference requires an input_data dict and additional **kwargs to update SamplingParams.
    Subclasses must implement:
      - get_engine_args(...) to provide engine configuration.
    Generation workers allow dynamic generation parameters (temperature, n, max tokens),
    while logprob workers enforce fixed parameters for single token generation.
    """
    def __init__(self, model_path: str, worker_id: str,
                 tensor_parallel_size: int = 1, max_num_seqs: int = 16):
        self.model_path = model_path
        self.worker_id = worker_id
        self.counter = 0
        print(f"Initializing {self.__class__.__name__} with model path: {model_path}")
        self.engine_args = self.get_engine_args(model_path, tensor_parallel_size, max_num_seqs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.llm = AsyncLLMEngine.from_engine_args(self.engine_args)
        atexit.register(lambda: asyncio.get_event_loop().create_task(self._async_cleanup()))
    
    def get_engine_args(self, model_path: str, tensor_parallel_size: int, max_num_seqs: int) -> AsyncEngineArgs:
        raise NotImplementedError("Subclasses must implement get_engine_args method.")
    
    def setup_registration(self):
        try:
            ray.get(self.registry.register.remote(service_id=self.worker_id))
            print(f"Worker {self.worker_id} registered.")
        except Exception as e:
            print(f"Error during registration for worker {self.worker_id}: {e}")
    
    async def _async_cleanup(self):
        try:
            await self.registry.deregister.remote(service_id=self.worker_id)
            print(f"Worker {self.worker_id} deregistered successfully.")
        except Exception as e:
            print(f"Error during async cleanup of worker {self.worker_id}: {e}")
    
    def update_weights(self, new_state_dict: dict):
        llm_model = self.llm.engine.model_executor.driver_worker.model_runner.model
        # for k,v in new_state_dict.items():
        #     v = v.to("cuda")
        #     llm_model.load_weights({k: v}.items())
        llm_model.load_weights(new_state_dict.items())
        print(f"vLLM weights updated successfully on service {self.worker_id}.")
        return True
    
    async def inference(self, input_data: dict, **kwargs) -> list[dict]:
        """
        Base inference:
          - Input: input_data must include 'input_token_ids'.
          - **kwargs can include extra parameters.
          - Constructs a SamplingParams object and then performs inference.
        """
        try:
            input_ids = input_data['input_token_ids']
            sampling_params = SamplingParams(
                **kwargs
            )
            request_id = sha256((str(sampling_params) + str(input_ids) + str(self.counter))
                                .encode('utf-8')).hexdigest()
            self.counter += 1
            print(f"[{self.__class__.__name__} {self.worker_id}] Sampling params: {sampling_params}")
            generator = self.llm.generate(
                prompt=TokensPrompt(prompt_token_ids=input_ids),
                sampling_params=sampling_params,
                request_id=request_id
            )
            result = None
            async for out in generator:
                result = out
            return result
        except Exception as e:
            print(f"Error during inference for worker {self.worker_id}: {e}")
            return None

@ray.remote
class GenerationVLLMWorker(BaseVLLMWorker):
    def __init__(self, *args, **kwargs):
        # Pass model_path, worker_id, tensor_parallel_size, max_num_seqs to Base
        super().__init__(*args, **kwargs)
        self.verify_worker = VerifierWorker.options(
            name=f"verifier_worker_{str(uuid.uuid4())}",
            num_cpus=4,
        ).remote()
        self.registry = get_or_create_registry("generation_vllm_registry")
        self.setup_registration()
    
    def get_engine_args(self, model_path: str, tensor_parallel_size: int, max_num_seqs: int) -> AsyncEngineArgs:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path)
        return AsyncEngineArgs(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.98,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            max_num_seqs=max_num_seqs,
            max_model_len=config.max_position_embeddings,
        )
    
    async def inference(self, sample: dict, max_new_tokens: int = None, **kwargs) -> list[dict]:
        # For generation, parameters are flexible.
        generation_kwargs = {
            "n": kwargs.get("n", 1),
            "max_tokens": max_new_tokens if max_new_tokens is not None \
                else self.engine_args.max_model_len - len(sample['input_token_ids']) - 1,
            "temperature": kwargs.get("temperature", 0.7),
            "include_stop_str_in_output": True,
            "spaces_between_special_tokens": False,
            "skip_special_tokens": False,
        }

        request_out = await super().inference(sample, **generation_kwargs)
        if 'input_len' not in sample:
            sample['input_len'] = len(sample['input_token_ids'])
        
        samples = [deepcopy(sample) for _ in range(len(request_out.outputs))]
        for sample, out in zip(samples, request_out.outputs):
            sample['output_token_ids'] = list(out.token_ids)
            sample['output_len'] = len(sample['output_token_ids'])
            sample['sample_ids'] = sample['input_token_ids'] + sample['output_token_ids']
            sample['sample_text'] = self.tokenizer.decode(sample['sample_ids'])
            sample['sample_position_ids'] = list(range(len(sample['sample_ids'])))
            sample['reward'] = await self.verify_worker.verify.remote(sample)
        
        group_rewards = np.array([s['reward'] for s in samples])
        group_advantages = normalize_rewards(group_rewards)
        for sample_, advantage in zip(samples, group_advantages):
            sample_['advantage'] = advantage.item()
        
        return samples

parsing_func = partial(parse, extraction_config=[
    LatexExtractionConfig(
        boxed_match_priority=0, 
        normalization_config=NormalizationConfig(
            boxed="last"
        )
    )
])

@ray.remote
class VerifierWorker:
    def verify(self, sample: dict) -> float:
        try:
            return float(verify(
                parsing_func(r'\boxed{'+sample['gt_answer']+'}'),
                parsing_func(sample['sample_text']),
            ))
        except Exception as e:
            print(f"Error during verification: {e}")
            return 0

if __name__ == "__main__":
    ray.init(address="auto", namespace="test")
    parser = argparse.ArgumentParser(description="Start vLLM worker service")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model weights")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of tensor parallel units")
    parser.add_argument("--max_num_seqs", type=int, default=16, help="Maximum number of sequences to generate")
    parser.add_argument("--mode", type=str, required=True, choices=["generation", "logprob"], help="Worker mode: generation or logprob")
    args = parser.parse_args()
    
    service_id = f"vllm_worker_{str(uuid.uuid4())}"
    if args.mode == "logprob":
        worker = LogprobVLLMWorker.options(
            name=service_id,
            num_gpus=args.tensor_parallel_size,
            num_cpus=4,
        ).remote(args.model_path, service_id, args.tensor_parallel_size, args.max_num_seqs)
    elif args.mode == "generation":
        worker = GenerationVLLMWorker.options(
            name=service_id,
            num_gpus=args.tensor_parallel_size,
            num_cpus=4,
            runtime_env={
                "pip": [f"vllm --extra-index-url https://wheels.vllm.ai/nightly"]
            },
        ).remote(args.model_path, service_id, args.tensor_parallel_size, args.max_num_seqs)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
    
    # Wait for the registry to be available.
    registry_name = "generation_vllm_registry" if args.mode == "generation" else "logprob_vllm_registry"
    while True:
        try:
            ray.get_actor(registry_name)
            print(f"{registry_name} found.")
            break
        except Exception:
            print(f"{registry_name} not found, sleeping for 1 second...")
            time.sleep(1)
    
    async def main():
        input_ids = [
            100264, 882, 100266, 4438, 1053, 499, 12849, 279, 8286, 315,
            264, 6211, 1903, 315, 279, 11552, 315, 1403, 66818, 315,
            279, 1890, 10801, 1405, 279, 19169, 72359, 449, 279, 7479,
            315, 279, 1023, 26436, 30, 100265, 100264, 78191, 100266
        ]
        actor = ray.get_actor(service_id)
        result1 = await actor.inference.remote({'input_token_ids': input_ids + [100264]})
        result2 = await actor.inference.remote({'input_token_ids': input_ids + [100264]})

        registry = get_or_create_registry(registry_name)
        tasks = [registry.inference_balanced.remote({'input_token_ids': input_ids}) for _ in range(2)]
        results = await asyncio.gather(*tasks)
        return result1, result2, results
    
    results = asyncio.run(main())
    print(results)
    
    while True:
        time.sleep(10000)

'''
Usage example:
mamba activate ray
for i in (seq 0 7)
    if test $i -lt 6
        set mode "generation"
        set max_num_seqs 8
        echo -e "\033[32mLaunching $mode actor on GPU $i with max_num_seqs $max_num_seqs\033[0m"
        env CUDA_VISIBLE_DEVICES="$i" python vllm_worker.py --model_path /dev/shm/phi-4 --mode $mode --tensor_parallel_size 1 --max_num_seqs $max_num_seqs &
    else
        # CUDA_VISIBLE_DEVICES="$i" python logprob_worker.py --model_path /dev/shm/phi-4 &
        CUDA_VISIBLE_DEVICES="$i" torchrun --nproc_per_node=1 --master_port=1234$i logprob_worker.py --model_path /dev/shm/phi-4 &
    end
end
set i 6
mamba activate grpo
CUDA_VISIBLE_DEVICES="$i" python logprob_worker.py --model_path /dev/shm/phi-4 &
torchrun --nproc_per_node=8 --master_port=12345 logprob_worker.py --model_path /dev/shm/phi-4
'''