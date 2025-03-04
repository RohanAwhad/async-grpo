from copy import deepcopy
import json
import logging
import random
import shutil
import time
import uuid
import asyncio
from functools import partial

import ray
from filelock import FileLock, Timeout
from wrapt_timeout_decorator import timeout
from deepscaler_math_utils import extract_answer, grade_answer_mathd, grade_answer_sympy
from utils import patch_target_module
from functools import partial
patch_target_module("math_verify.utils.timeout", partial(timeout, use_signals=False))
# Import verification functions from math_verify
from math_verify import verify, parse
from math_verify.parser import LatexExtractionConfig, NormalizationConfig
import re

import numpy as np

def cos_fn(t, T, eta_min, eta_max):
    """Basic cosine function component"""
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(t * np.pi / T))


def compute_cosine_reward(gen_length, max_length, format_quality, is_correct, 
                          r_c_0=6, r_c_L=5, r_w_0=-10, r_w_L=0, r_f_0=1.0, r_f_L=0.5,
                          exceed_penalty=-10):
    """
    Modification of the cosine reward function from this paper to include format quality.
    'Demystifying Long Chain-of-Thought Reasoning in LLMs' (Yeo et al., 2025)
    arXiv:2502.03373
      
    Parameters:
    gen_length: Generation length
    max_length: Maximum allowed length
    format_quality: 1=correct format, 0=incorrect format (None uses is_correct only)
    is_correct: 1=correct answer, 0=incorrect answer
    r_c_0/r_c_L: Rewards for correct at min/max length
    r_w_0/r_w_L: Rewards for wrong at min/max length
    r_f_0/r_f_L: Rewards for wrong but good format at min/max length
    exceed_penalty: Penalty for exceeding max length
    """
    # Check if generation length exceeds maximum length
    if gen_length >= max_length:
        return exceed_penalty
    
    if is_correct == 1:
        reward = cos_fn(gen_length, max_length, r_c_0, r_c_L)
    else:
        reward = cos_fn(gen_length, max_length, r_w_0, r_w_L)

    reward += cos_fn(gen_length, max_length, r_f_0, r_f_L) if format_quality == 1 else 0
    
    return reward

@ray.remote
class VerifierWorker:
    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        print(f"Initializing VerifierWorker with id: {worker_id}")

    def extract_reference_and_answer(self, sample: dict):
        original_input = sample['input']
        output = sample['sample_text'].split(original_input)[1]
        if "\\boxed" in sample['answer']:
            sample['parsed_gt_answer'] = extract_answer(sample['answer'])
        else:
            sample['parsed_gt_answer'] = sample['answer']
        try:
            sample['parsed_attempt'] = extract_answer(output)
        except Exception:
            sample['parsed_attempt'] = ''
        return sample
    
    def verify_mathd(self, sample: dict, max_gen_length: int):
        sample = self.extract_reference_and_answer(sample)
        sample['original_reward'] = grade_answer_mathd(sample['parsed_attempt'], sample['parsed_gt_answer'])
        sample['reward'] = compute_cosine_reward(sample['output_len'], 
                                                 max_gen_length, 
                                                 format_quality=sample['parsed_attempt'] != '', 
                                                 is_correct=sample['original_reward'])
        return sample
    
    def verify_sympy(self, sample: dict, max_gen_length: int):
        sample = self.extract_reference_and_answer(sample)
        sample['original_reward'] = grade_answer_sympy(sample['parsed_attempt'], sample['parsed_gt_answer'])
        sample['reward'] = compute_cosine_reward(sample['output_len'], 
                                                 max_gen_length, 
                                                 format_quality=sample['parsed_attempt'] != '', 
                                                 is_correct=sample['original_reward'])
        return sample

@ray.remote
class VerifierPool:
    def __init__(self, global_num_verifiers: int, write_failed: bool = False):
        self.node_id = ray.get_runtime_context().get_node_id()
        self.global_num_verifiers = global_num_verifiers
        self.write_failed = write_failed
        self.verifier_pool = [None for _ in range(global_num_verifiers)]
        self.verifier_load = [0 for _ in range(global_num_verifiers)]
        self.lock = asyncio.Lock()
        self.create_verifier_tasks = [asyncio.create_task(self.create_verifier(i)) for i in range(global_num_verifiers)]
        shutil.rmtree(f"failed_samples_verify.jsonl", ignore_errors=True)
        shutil.rmtree(f"failed_samples_verify.jsonl.lock", ignore_errors=True)
        self.time_since_last_failed = time.time()

    async def create_verifier(self, index: int):
        async with self.lock:
            self.verifier_pool[index] = VerifierWorker.options(
                num_cpus=1, 
                scheduling_strategy="SPREAD",
            ).remote(f"verifier_{index}_{str(uuid.uuid4())}", self.write_failed)
            self.verifier_load[index] = 0

    async def write_failed_sample(self, sample: dict):
        print("\033[38;5;196m\033[1m DEBUG: Failed to verify sample \033[0m", flush=True)
        if self.write_failed:
            try:
                with FileLock(f"failed_samples_verif.jsonl.lock", timeout=20):
                    with open(f"failed_samples_verify.jsonl", "a") as f:
                        f.write(json.dumps(sample) + "\n")
            except Timeout:
                print("Lock acquisition failed after 20 seconds", flush=True)
        return sample
    
    async def _verify_balanced(self, sample: dict, mode: str, **kwargs) -> dict:
        result = deepcopy(sample)
        # result[f'reward_{mode}'] = 0.0
        result['reward'] = 0.0
        for _ in range(2):
            try:
                async with self.lock:
                    min_index = min(range(len(self.verifier_load)), key=lambda i: self.verifier_load[i])
                    self.verifier_load[min_index] += 1
                if mode == 'mathd':
                    result_ref = self.verifier_pool[min_index].verify_mathd.remote(result, **kwargs)
                elif mode == 'sympy':
                    result_ref = self.verifier_pool[min_index].verify_sympy.remote(result, **kwargs)
                else:
                    raise ValueError(f"Invalid mode: {mode}")
                result =  await asyncio.wait_for(result_ref, 30)
                async with self.lock:
                    self.verifier_load[min_index] -= 1
                break
            except Exception as e:
                if time.time() - self.time_since_last_failed > 60:
                    import traceback
                    traceback.print_exc()
                    print(f"\033[1;38;5;196mCoroutine died in verify_balanced\033[0m", flush=True)
                    print(f"\033[1;38;5;196mSample Text: \033[0m {sample['sample_text'][-100:]}", flush=True)
                    print(f"\033[1;38;5;196mSample Answer: \033[0m {sample['answer']}", flush=True)
                    async with self.lock:
                        self.time_since_last_failed = time.time()
                # raise e
                await self.create_verifier(min_index)
                await self.write_failed_sample(result)
                await asyncio.sleep(random.uniform(0.1, 5))
        return result
    
    def pick_verified_sample(self, results: list[dict]) -> dict:
        '''
        Pick the first verified sample that is correct. or the first verified sample that was parsed.
        '''
        for result in results:
            if result['original_reward']:
                return result
        for result in results:
            if result['parsed_attempt'] != '':
                return result
        return results[0]
 
    async def verify_balanced(self, sample: dict, **kwargs) -> dict:
        tasks = [asyncio.create_task(self._verify_balanced(sample, 'mathd', **kwargs)),
                 asyncio.create_task(self._verify_balanced(sample, 'sympy', **kwargs))]
        results = await asyncio.gather(*tasks)
        return self.pick_verified_sample(results)


def get_or_create_verifier_pool(global_num_verifiers: int, write_failed: bool = False) -> VerifierPool:
    # For simplicity, always create a new instance. In a production setting, you might want to implement a singleton.
    try:
        return VerifierPool.options(name="verifier_pool").remote(global_num_verifiers, write_failed) 
    except Exception as e:
        return ray.get_actor("verifier_pool")
