import asyncio
import os
import time
import logging
import ray

import re
import random
from enum import Enum
from dataclasses import dataclass
from typing import Any, Optional

DEBUG = False
def debug_log(message: str, file_name: str = None):
    if DEBUG:
        print(message)
        # compute path inside the worker's environment
        path = file_name or os.path.join(os.getcwd(), "debug_vllm_experience_batcher.txt")
        with open(path, "a") as _f:
            _f.write(message + "\n")

async def get_experience_and_ref_logprobs(sample, num_samples, actor_registry_handle, reference_registry_handle, temperature=1.0, max_tokens=8192, insert_reasoning_phrases=False):
    
    debug_log(f"Getting experience and reference logprobs for sample: {sample['input']}")
    # actor_registry = ray.get_actor(actor_registry_name)
    samples = await actor_registry_handle.inference_balanced.remote(
        sample,
        n=num_samples,
        temperature=temperature,
        max_tokens=max_tokens,
        insert_reasoning_phrases=insert_reasoning_phrases
    )
    debug_log(f"Samples for {sample['input']}")
    # logging.debug(f"\033[1;38;2;255;165;0mFirst sample before rewriting: \033[0m {samples[0]['sample_text']}")
    # samples = await asyncio.gather(*[rewrite_with_insert_phrase(s) for s in samples])
    
    # logging.debug(f"\033[1;38;2;255;165;0mFirst sample after rewriting: \033[0m {samples[0]['input']}")
    # samples = await asyncio.gather(*[
    #     actor_registry.inference_balanced.remote(s, n=1, temperature=temperature) 
    #     for s in samples])
    
    # logging.debug(f"\033[1;38;2;255;165;0mFirst sample after rewriting: \033[0m {samples[0]['sample_text']}")
    # reference_registry = ray.get_actor(reference_registry_name)

    tasks = [reference_registry_handle.inference_balanced.remote(
        s,
    ) for s in samples]
    samples_with_ref_logprobs = await asyncio.gather(*tasks)
    debug_log(f"Samples with reference logprobs for {sample['input']}")
    return samples_with_ref_logprobs

class MessageType(Enum):
    MINIBATCH = "minibatch"
    GRADIENT_STEP = "gradient_step"
    BATCH_DONE = "batch_done"

@dataclass
class Message:
    type: MessageType
    data: Optional[Any] = None

@ray.remote
class ExperienceBatcher:
    def __init__(self):
        self.training_processes_queues = {}
        self.training_batches = {}
        self.training_batches_lengths = {}
        self.experience_queue = []
        self.ready_experience_samples = []
        self.actor_registry_handle = ray.get_actor("generation_vllm_registry")
        self.reference_registry_handle = ray.get_actor("logprob_vllm_registry")
        self.lock = asyncio.Lock()
        self.dispatched_since_last_grad_step = 0
        self.train_minibatch_size = None
    
    def start_creating_batches(self):
        asyncio.create_task(self._create_batches())

    def register_training_process(self, global_rank: int, max_tokens_per_gpu: int, train_minibatch_size: int):
        self.max_tokens_per_gpu = max_tokens_per_gpu
        self.train_minibatch_size = train_minibatch_size
        self.training_processes_queues[global_rank] = asyncio.Queue()
        self.training_batches[global_rank] = []
        self.training_batches_lengths[global_rank] = 0
        return self.training_processes_queues[global_rank]
    

    async def add_sample_to_batches(self, sample):
        # Compute sample length and pick the batch to fill
        sample_len = sample['input_len'] + sample['output_len']
        least_id = min(self.training_batches_lengths, key=self.training_batches_lengths.get)
        # If adding would overflow tokens, flush existing batches first
        if self.training_batches_lengths[least_id] + sample_len > self.max_tokens_per_gpu:
            num_dispatched = await self.dispatch_batches()
            if num_dispatched == 0:
                raise Exception("Token overflow with no batches to dispatch")
            self.dispatched_since_last_grad_step += num_dispatched
        # Add the new sample
        self.training_batches[least_id].append(sample)
        self.training_batches_lengths[least_id] += sample_len
        # If we've exactly hit the train_minibatch_size, flush and trigger gradient
        if self.dispatched_since_last_grad_step + sum([len(batch) for batch in self.training_batches.values()]) == self.train_minibatch_size:
            num_dispatched = await self.dispatch_batches()
            if not num_dispatched:
                raise Exception("Reached minibatch size with no batches to dispatch")
            await self.dispatch_signal(MessageType.GRADIENT_STEP)
            self.dispatched_since_last_grad_step = 0
    
    async def reset_batches(self):
        for batch_id in self.training_batches:
            self.training_batches[batch_id] = []
            self.training_batches_lengths[batch_id] = 0
    
    async def dispatch_batches(self):
        """
        Emit exactly one MINIBATCH to each worker (with dummy fill for empty queues)
        only if at least one batch contains real samples. Return True if sent.
        """
        # do nothing if no real samples anywhere
        num_real_samples = 0
        if not any(self.training_batches.values()):
            return num_real_samples

        dummy_sample = {'dummy': True}
        dispatch_tasks = []
        for batch_id, batch in self.training_batches.items():
            payload = batch if batch else [dummy_sample]
            dispatch_tasks.append(self.training_processes_queues[batch_id].put(Message(MessageType.MINIBATCH, payload)))
            num_real_samples += len(batch) if batch else 0
        await asyncio.gather(*dispatch_tasks)
        await self.reset_batches()
        return num_real_samples

    async def dispatch_signal(self, signal_type: MessageType):
        """Send a signal to all workers."""
        for queue in self.training_processes_queues.values():
            await queue.put(Message(signal_type))


    async def _create_batches(self):
        """Continuously consumes tasks from the experience_queue and processes them."""
        async with self.lock:
            for task in asyncio.as_completed(self.experience_queue):
                debug_log(f"Experience queue length in _create_batches: {len(self.experience_queue)}")
                samples = await task
                if samples is None: # underlying coroutine timed out
                    continue
                for sample in samples:
                    await self.add_sample_to_batches(sample)
            debug_log(f"Experience queue length in _create_batches after processing: {len(self.experience_queue)}")
            self.experience_queue = []
        
            num_dispatched = await self.dispatch_batches()
            if num_dispatched > 0:
                await self.dispatch_signal(MessageType.GRADIENT_STEP)
            await self.dispatch_signal(MessageType.BATCH_DONE)
            debug_log("Batch done dispatched")

    async def get_batch(self, global_rank: int):
        return await self.training_processes_queues[global_rank].get()
    
    async def generate_experience(self, 
                                  samples, 
                                  samples_per_question, 
                                  actor_registry="generation_vllm_registry", 
                                  reference_registry="logprob_vllm_registry",
                                  temperature=1.0,
                                  max_tokens=8192,
                                  timeout=600,
                                  insert_reasoning_phrases=False):
        """
        Asynchronously processes a batch of questions to generate and accumulate samples while 
        ensuring that each accumulated batch does not exceed a specified token limit.

        For each question, the function asynchronously obtains multiple samples (along with their
        reference log probabilities). It then computes normalized rewards (advantages) for each sample
        and attaches them. Samples are accumulated until the total token count (input length plus output 
        length) exceeds the `max_tokens_per_gpu`. When the limit is exceeded, the accumulated samples
        are post-processed via `post_process_batch` and yielded as one batch. Any remaining samples
        after processing all questions are also post-processed and yielded.

        Args:
            samples (iterable): A collection of samples to be processed.
            samples_per_question (int): The number of samples to generate per question.
            max_tokens_per_gpu (int): Maximum allowed cumulative token count for each yielded batch.
            device (torch.device): The device to which the batch is transferred during post-processing.
            actor_registry (str, optional): Name of the actor registry for sample generation. 
                                            Defaults to "generation_vllm_registry".
            reference_registry (str, optional): Name of the actor registry for obtaining reference
                                                log probabilities. Defaults to "logprob_vllm_registry".

        Yields:
            dict: A post-processed batch of samples that satisfies the maximum token constraint.
        """
        # Initiate asynchronous requests for getting experience and reference logprobs.
        print(f"\033[1;32mStarting to get experience and reference logprobs for {len(samples)} samples\033[0m")
        tasks_samples_with_ref_logprobs = [
            get_experience_and_ref_logprobs(
                sample,
                samples_per_question,
                self.actor_registry_handle,
                self.reference_registry_handle,
                temperature=temperature,
                max_tokens=max_tokens,
                insert_reasoning_phrases=insert_reasoning_phrases
            ) for sample in samples
        ]
        async with self.lock:
            wrapped_tasks = [asyncio.create_task(self.run_with_timeout(t, timeout))
                             for t in tasks_samples_with_ref_logprobs]
            self.experience_queue.extend(wrapped_tasks)

        return True
    
    async def run_with_timeout(self, coroutine, timeout=600):
        try:
            return await asyncio.wait_for(coroutine, timeout)
        except asyncio.TimeoutError:
            logging.warning(f"Coroutine {coroutine} timed out after {timeout} seconds")
            return None
        except Exception as e:
            logging.warning(f"Error running coroutine {coroutine}: {e}")
            return None

def get_or_create_experience_batcher(experience_batcher_name: str):
    try:
        return ExperienceBatcher.options(
            name=experience_batcher_name,
            num_cpus=8,
            namespace="test",
            runtime_env={"env_vars": dict(os.environ),
                        "pip": [f"-r {os.path.join(os.path.dirname(__file__), 'requirements_fsdp.txt')}"]
                        }
        ).remote()
    except Exception:
        return ray.get_actor(experience_batcher_name, namespace="test")
    

if __name__ == "__main__":
    ray.init(address="auto", namespace="test")
    # batcher = get_or_create_experience_batcher("experience_batcher")
    batcher = ExperienceBatcher.options(
        name="experience_batcher",
        num_cpus=8,
        namespace="test",
    ).remote()
    time.sleep(10)
    ray.get(batcher.register_training_process.remote(0, 25000, 100000))
    ray.get(batcher.generate_experience.remote([{'input_token_ids': [100264, 882, 100266, 4438, 1053, 499, 12849, 279, 8286, 315,
            264, 6211, 1903, 315, 279, 11552, 315, 1403, 66818, 315,
            279, 1890, 10801, 1405, 279, 19169, 72359, 449, 279, 7479,
            315, 279, 1023, 26436, 30, 100265, 100264, 78191, 100266]}], 4))
    ray.get(batcher.start_consuming_experience.remote())
    print(ray.get(batcher.get_batch.remote(0)))



