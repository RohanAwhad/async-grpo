import argparse
from copy import deepcopy
from functools import partial
from hashlib import sha256
import json
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
from vllm_registry import get_or_create_registry 
import asyncio
# from math_verify import parse, verify
# from math_verify.parser import LatexExtractionConfig, NormalizationConfig
import re
# from verifier_registry import get_or_create_verifier_registry, verify_single
# from math_verify import parse, verify  # use basic parsing and verification

from filelock import FileLock, Timeout

import numpy as np
from numba import njit
logging.getLogger('numba').setLevel(logging.WARNING)

from utils import patch_target_module
from wrapt_timeout_decorator import timeout
# need to patch since we are using the module inside ray. see: 
patch_target_module("math_verify.utils.timeout", partial(timeout, use_signals=False))
from math_verify import verify
from math_verify.parser import LatexExtractionConfig, NormalizationConfig
from math_verify import parse

def verify_sample(sample: dict) -> float:
    parsing_func_ = partial(
        parse,
        extraction_config=[
            LatexExtractionConfig(
                try_extract_without_anchor=False,
                boxed_match_priority=0, 
                normalization_config=NormalizationConfig(
                    boxed="last"
                )
            )
        ],
        fallback_mode="no_fallback",
        extraction_mode="first_match",
        parsing_timeout=20,
    )
    parsed_gt_answer = parsing_func_(r'\boxed{' + sample['gt_answer'] + '}')
    parsed_attempt = parsing_func_(sample['sample_text'])
    # Run verification.
    result = float(verify(
        parsed_gt_answer,
        parsed_attempt,
        timeout_seconds=20,
    ))
    # gt_answer_parsed = parsing_func_(r'\boxed{' + sample['gt_answer'] + '}')
    # sample_text_parsed = parsing_func_(sample['sample_text'])
    # print(f"\033[1;38;2;255;165;0mDEBUG Verifier sample text: \033[0m{sample['sample_text']}\033[1;38;2;255;165;0m with ground truth: \033[0m{sample['gt_answer']}\033[1;38;2;255;165;0m and result: \033[0m{result}", flush=True)
    # print(f"\033[1;38;2;255;165;0mDEBUG ground truth: \033[0m{gt_answer_parsed}\033[1;38;2;255;165;0m with sample text: \033[0m{sample_text_parsed}")
    sample['reward'] = result
    sample['parsed_gt_answer'] = parsed_gt_answer
    sample['parsed_attempt'] = parsed_attempt
    return sample

@ray.remote
class VerifyWorker:
    def __init__(self, worker_id: str, write_failed: bool = False):
        self.worker_id = worker_id
        self.write_failed = write_failed
        print(f"Initializing VerifyWorker with id: {worker_id}, write_failed: {self.write_failed}")
    
    def verify_sample(self, sample: dict) -> float:
        for _ in range(3):
            try:
                return timeout(30, use_signals=False, exception_message="")(verify_sample)(sample)
            except Exception as e:
                time.sleep(0.1)
        print(f"\033[38;5;196m\033[1m DEBUG:Failed to verify \033[0m\n", flush=True)
        sample['reward'] = 0.0
        if self.write_failed:
            try:
                with FileLock("failed_generation_samples.jsonl.lock", timeout=20):
                    with open("failed_generation_samples.jsonl", "a") as f:
                        f.write(json.dumps(sample) + "\n")
            except Timeout:
                print("Lock acquisition failed after 20 seconds", flush=True)
        return sample

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
            print(f"\033[38;5;196m\033[1mError during inference for worker {self.worker_id}: {e}\033[0m")
            import traceback
            traceback.print_exc()
            raise e

@ray.remote
class GenerationVLLMWorker(BaseVLLMWorker):
    def __init__(self, model_path: str, worker_id: str, tensor_parallel_size: int, max_num_seqs: int,
                 num_verifiers: int = 4, write_failed: bool = False):
        # Pass the common parameters to the base initializer.
        super().__init__(model_path, worker_id, tensor_parallel_size, max_num_seqs)
        # Create verifier workers while propagating the write_failed flag.
        self.verifier_pool = [
            VerifyWorker.options(num_cpus=1).remote(f"verifier_{i}_{str(uuid.uuid4())}", write_failed)
            for i in range(num_verifiers)
        ]
        self.verifier_load = [0] * num_verifiers
        self.verifier_lock = asyncio.Lock()
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
    
    async def verify_balanced(self, sample: dict) -> dict:
        # Acquire the lock to safely choose the least busy verifier.
        async with self.verifier_lock:
            min_index = min(range(len(self.verifier_load)), key=lambda i: self.verifier_load[i])
            self.verifier_load[min_index] += 1
            
        result_ref = self.verifier_pool[min_index].verify_sample.remote(sample)
        result = await result_ref

        async with self.verifier_lock:
            self.verifier_load[min_index] -= 1
        return result
    
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
        
        sample_rewards_futures = []
        for sample, out in zip(samples, request_out.outputs):
            sample['output_token_ids'] = list(out.token_ids)
            sample['output_len'] = len(sample['output_token_ids'])
            sample['sample_ids'] = sample['input_token_ids'] + sample['output_token_ids']
            sample['sample_text'] = self.tokenizer.decode(sample['sample_ids'])
            sample['sample_position_ids'] = list(range(len(sample['sample_ids'])))
            sample_rewards_futures.append(self.verify_balanced(sample))

        samples = await asyncio.gather(*sample_rewards_futures)
        group_rewards = np.array([s['reward'] for s in samples])
        group_advantages = normalize_rewards(group_rewards)
        for sample_, advantage in zip(samples, group_advantages):
            sample_['advantage'] = advantage.item()
        
        print(f"\033[38;5;201mWorker \033[0m {self.worker_id} \033[38;5;201mfinished inference with \033[0m {len(samples)} samples.")
        return samples


if __name__ == "__main__":
    ray.init(address="auto", namespace="test")
    parser = argparse.ArgumentParser(description="Start vLLM worker service")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model weights")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of tensor parallel units")
    parser.add_argument("--max_num_seqs", type=int, default=16, help="Maximum number of sequences to generate")
    parser.add_argument("--mode", type=str, required=True, choices=["generation", "logprob"], help="Worker mode: generation or logprob")
    parser.add_argument("--num_verifiers", type=int, default=4,
                        help="Number of verifier workers")
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
        ).remote(
            args.model_path,
            service_id,
            args.tensor_parallel_size,
            args.max_num_seqs,
            args.num_verifiers,
            args.write_failed
        )
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