from copy import deepcopy
import json
import logging
from pathlib import Path
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
logging.getLogger().setLevel(logging.DEBUG)


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
    
    return reward.item()

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
    
    def verify_both(self, sample: dict, max_gen_length: int):
        sample = self.extract_reference_and_answer(sample)
        sample['original_reward'] = grade_answer_mathd(sample['parsed_attempt'], sample['parsed_gt_answer']) or grade_answer_sympy(sample['parsed_attempt'], sample['parsed_gt_answer'])
        # TODO: Add cosine reward
        sample['reward'] = sample['original_reward']
        return sample
    
    def verify_mathd(self, sample: dict, max_gen_length: int):
        sample = self.extract_reference_and_answer(sample)
        sample['original_reward'] = grade_answer_mathd(sample['parsed_attempt'], sample['parsed_gt_answer'])
        sample['reward'] = sample['original_reward']
        # sample['reward'] = compute_cosine_reward(sample['output_len'], 
        #                                          max_gen_length, 
        #                                          format_quality=sample['parsed_attempt'] != '', 
        #                                          is_correct=sample['original_reward'])
        return sample
    
    def verify_sympy(self, sample: dict, max_gen_length: int):
        sample = self.extract_reference_and_answer(sample)
        sample['original_reward'] = grade_answer_sympy(sample['parsed_attempt'], sample['parsed_gt_answer'])
        sample['reward'] = sample['original_reward']
        # sample['reward'] = compute_cosine_reward(sample['output_len'], 
        #                                          max_gen_length, 
        #                                          format_quality=sample['parsed_attempt'] != '', 
        #                                          is_correct=sample['original_reward'])
        return sample

@ray.remote
class VerifierPool:
    def __init__(self, global_num_verifiers: int, write_failed: bool = False, output_dir: str = None):
        self.node_id = ray.get_runtime_context().get_node_id()
        self.global_num_verifiers = global_num_verifiers
        self.write_failed = write_failed
        self.lock = asyncio.Lock()
        # Split the global verifiers between mathd and sympy
        n_mathd = global_num_verifiers // 2
        n_sympy = global_num_verifiers - n_mathd
        self.verifier_pool_mathd = [None for _ in range(n_mathd)]
        self.verifier_load_mathd = [0 for _ in range(n_mathd)]
        self.verifier_pool_sympy = [None for _ in range(n_sympy)]
        self.verifier_load_sympy = [0 for _ in range(n_sympy)]
        self.create_verifier_tasks = ([asyncio.create_task(self.create_verifier_mathd(i)) for i in range(n_mathd)] +
                                       [asyncio.create_task(self.create_verifier_sympy(i)) for i in range(n_sympy)])
        self.time_since_last_failed = time.time()
        self.outfile = Path(output_dir) / "failed_samples_verify.jsonl" if output_dir is not None else Path("failed_samples_verify.jsonl")
        self.outfile.unlink(missing_ok=True)
        Path(str(self.outfile)+'.lock').unlink(missing_ok=True)
        
    async def create_verifier_mathd(self, index: int):
        async with self.lock:
            self.verifier_pool_mathd[index] = VerifierWorker.options(
                num_cpus=1, 
                scheduling_strategy="SPREAD",
            ).remote(f"verifier_mathd_{index}_{str(uuid.uuid4())}")
            self.verifier_load_mathd[index] = 0

    async def create_verifier_sympy(self, index: int):
        async with self.lock:
            self.verifier_pool_sympy[index] = VerifierWorker.options(
                num_cpus=1, 
                scheduling_strategy="SPREAD",
            ).remote(f"verifier_sympy_{index}_{str(uuid.uuid4())}")
            self.verifier_load_sympy[index] = 0

    async def write_failed_sample(self, sample: dict):
        print("\033[38;5;196m\033[1m DEBUG: Failed to verify sample \033[0m", flush=True)
        if self.write_failed:
            try:
                with FileLock(f"{self.outfile}.lock", timeout=20):
                    with open(self.outfile, "a") as f:
                        f.write(json.dumps(sample) + "\n")
            except Timeout:
                print("Lock acquisition failed after 20 seconds", flush=True)
        return sample
    
    async def _verify_single(self, sample: dict, mode: str, **kwargs) -> dict:
        result = sample
        for _ in range(2):
            try:
                if mode == 'mathd':
                    async with self.lock:
                        min_index = min(range(len(self.verifier_load_mathd)), key=lambda i: self.verifier_load_mathd[i])
                        self.verifier_load_mathd[min_index] += 1
                    result_ref = self.verifier_pool_mathd[min_index].verify_mathd.remote(sample, kwargs['max_gen_length'])
                    result = await asyncio.wait_for(result_ref, 30)
                    async with self.lock:
                        self.verifier_load_mathd[min_index] -= 1
                elif mode == 'sympy':
                    async with self.lock:
                        min_index = min(range(len(self.verifier_load_sympy)), key=lambda i: self.verifier_load_sympy[i])
                        self.verifier_load_sympy[min_index] += 1
                    result_ref = self.verifier_pool_sympy[min_index].verify_sympy.remote(sample, kwargs['max_gen_length'])
                    result = await asyncio.wait_for(result_ref, 30)
                    async with self.lock:
                        self.verifier_load_sympy[min_index] -= 1
                else:
                    raise ValueError(f"Invalid mode: {mode}")
                # break
            except Exception as e:
                if mode == 'mathd':
                    await self.create_verifier_mathd(min_index)
                else:
                    await self.create_verifier_sympy(min_index)
                await asyncio.sleep(random.uniform(0.01, 0.05))
        return result
    
    async def pick_verified_sample(self, results: list[dict]) -> dict:
        # Choose the result with the highest reward or non '' parsed_attempt
        for result in results:
            if result['original_reward'] > 0:
                return result
        for result in results:
            if result['parsed_attempt'] != '':
                return result
        await self.write_failed_sample(results[0])
        return results[0]

    async def verify_balanced(self, sample: dict, **kwargs) -> dict:
        # Run both mathd and sympy verification tasks concurrently
        sample['original_reward'] = 0.0
        sample['reward'] = 0.0
        sample['parsed_attempt'] = ''
        tasks = [asyncio.create_task(self._verify_single(deepcopy(sample), 'mathd', **kwargs)),
                 asyncio.create_task(self._verify_single(deepcopy(sample), 'sympy', **kwargs))]
        results = await asyncio.gather(*tasks)
        return await self.pick_verified_sample(results)


def get_or_create_verifier_pool(global_num_verifiers: int, write_failed: bool = False) -> VerifierPool:
    # For simplicity, always create a new instance. In a production setting, you might want to implement a singleton.
    try:
        return VerifierPool.options(name="verifier_pool").remote(global_num_verifiers, write_failed) 
    except Exception as e:
        return ray.get_actor("verifier_pool")
