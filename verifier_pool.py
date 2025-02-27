import json
import time
import uuid
import asyncio
from functools import partial

import ray
from filelock import FileLock, Timeout
from wrapt_timeout_decorator import timeout
from utils import patch_target_module
from functools import partial
patch_target_module("math_verify.utils.timeout", partial(timeout, use_signals=False))
# Import verification functions from math_verify
from math_verify import verify, parse
from math_verify.parser import LatexExtractionConfig, NormalizationConfig


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
    result = float(verify(parsed_gt_answer, parsed_attempt, timeout_seconds=20))
    sample['reward'] = result
    sample['parsed_gt_answer'] = parsed_gt_answer
    sample['parsed_attempt'] = parsed_attempt
    return sample


@ray.remote
class VerifierWorker:
    def __init__(self, worker_id: str, write_failed: bool = False):
        self.worker_id = worker_id
        self.write_failed = write_failed
        print(f"Initializing VerifierWorker with id: {worker_id}, write_failed: {self.write_failed}")
    
    def verify_sample(self, sample: dict):
        for _ in range(3):
            try:
                # Wrap the verify_sample function with a timeout
                return timeout(30, use_signals=False, exception_message="")(verify_sample)(sample)
            except Exception as e:
                time.sleep(0.1)
        print("\033[38;5;196m\033[1m DEBUG: Failed to verify sample \033[0m", flush=True)
        sample['reward'] = 0.0
        if self.write_failed:
            try:
                with FileLock("failed_generation_samples.jsonl.lock", timeout=20):
                    with open("failed_generation_samples.jsonl", "a") as f:
                        f.write(json.dumps(sample) + "\n")
            except Timeout:
                print("Lock acquisition failed after 20 seconds", flush=True)
        return sample

@ray.remote
class VerifierPool:
    def __init__(self, global_num_verifiers: int, write_failed: bool = False):
        self.global_num_verifiers = global_num_verifiers
        self.write_failed = write_failed
        self.verifier_pool = [VerifierWorker.options(num_cpus=1).remote(f"verifier_{i}_{str(uuid.uuid4())}", write_failed)
                               for i in range(global_num_verifiers)]
        self.verifier_load = [0 for _ in range(global_num_verifiers)]
        self.lock = asyncio.Lock()

    async def verify_balanced(self, sample: dict) -> dict:
        async with self.lock:
            min_index = min(range(len(self.verifier_load)), key=lambda i: self.verifier_load[i])
            self.verifier_load[min_index] += 1
        result_ref = self.verifier_pool[min_index].verify_sample.remote(sample)
        result = await result_ref
        async with self.lock:
            self.verifier_load[min_index] -= 1
        return result


def get_or_create_verifier_pool(global_num_verifiers: int, write_failed: bool = False) -> VerifierPool:
    # For simplicity, always create a new instance. In a production setting, you might want to implement a singleton.
    try:
        return VerifierPool.options(name="verifier_pool").remote(global_num_verifiers, write_failed) 
    except Exception as e:
        return ray.get_actor("verifier_pool")
