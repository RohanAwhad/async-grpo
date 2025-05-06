import asyncio
import os
import time
import logging
import ray

import re
import random

DEBUG = True
# placeholder for debug file handle; will be opened inside the remote actor
debug_file = None

async def get_experience_and_ref_logprobs(sample, num_samples, actor_registry_handle, reference_registry_handle, temperature=1.0, max_tokens=8192, insert_reasoning_phrases=False):
    if DEBUG:
        debug_file.write(f"Getting experience and reference logprobs for sample: {sample['input']}\n")
        debug_file.flush()
    # actor_registry = ray.get_actor(actor_registry_name)
    samples = await actor_registry_handle.inference_balanced.remote(
        sample,
        n=num_samples,
        temperature=temperature,
        max_tokens=max_tokens,
        insert_reasoning_phrases=insert_reasoning_phrases
    )
    if DEBUG:
        debug_file.write(f"Samples for {sample['input']}\n")
        debug_file.flush()
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
    if DEBUG:
        debug_file.write(f"Samples with reference logprobs for {sample['input']}\n")
        debug_file.flush()
    return samples_with_ref_logprobs

@ray.remote
class ExperienceBatcher:
    def __init__(self):
        # open debug file in the actor process
        global debug_file
        log_path = os.path.join(os.path.dirname(__file__), "debug_vllm_experience_batcher.txt")
        debug_file = open(log_path, "a") if DEBUG else None
        self.training_processes_queues = {}
        self.training_batches = {}
        self.training_batches_lengths = {}
        self.experience_queue = []
        self.ready_experience_samples = []
        self.actor_registry_handle = ray.get_actor("generation_vllm_registry")
        self.reference_registry_handle = ray.get_actor("logprob_vllm_registry")
        self.lock = asyncio.Lock()
    
    def start_creating_batches(self):
        asyncio.create_task(self._create_batches())


    def register_training_process(self, global_rank: int, max_tokens_per_gpu: int):
        self.max_tokens_per_gpu = max_tokens_per_gpu
        self.training_processes_queues[global_rank] = asyncio.Queue()
        self.training_batches[global_rank] = []
        self.training_batches_lengths[global_rank] = 0
        return self.training_processes_queues[global_rank]
    

    async def add_sample_to_batches(self, sample):
        sample_len = sample['input_len'] + sample['output_len']
        least_full_batch_id = min(self.training_batches_lengths, key=self.training_batches_lengths.get)
        if self.training_batches_lengths[least_full_batch_id] + sample_len > self.max_tokens_per_gpu:
        #   or all(len(batch) > 1 for batch in self.training_batches.values()):
            dispatched = await self.dispatch_batches()
            if not dispatched:
                raise Exception("Didn't dispatch but can't add sample to batch because it exceeds max tokens per gpu")
        self.training_batches[least_full_batch_id].append(sample)
        self.training_batches_lengths[least_full_batch_id] += sample['input_len'] + sample['output_len']
    
    async def reset_batches(self):
        for batch_id in self.training_batches:
            self.training_batches[batch_id] = []
            self.training_batches_lengths[batch_id] = 0
    
    async def dispatch_batches(self):
        if all(len(batch) > 0 for batch in self.training_batches.values()):
            for batch_id, batch in self.training_batches.items():
                await self.training_processes_queues[batch_id].put(batch)
            await self.reset_batches()
            return True
        else:
            print(f"\033[1;31mCan't dispatch because there are {sum(True for batch in self.training_batches.values() if len(batch) > 0)} batches with samples out of {len(self.training_batches)} batches\033[0m")
            return False

    async def dispatch_sentinel(self):
        for queue in self.training_processes_queues.values():
            await queue.put(None)

    async def _create_batches(self):
        """Continuously consumes tasks from the experience_queue and processes them."""
        async with self.lock:
            for task in asyncio.as_completed(self.experience_queue):
                debug_file.write(f"Experience queue length in _create_batches: {len(self.experience_queue)}\n")
                debug_file.flush()
                samples = await task
                if samples is None: # underlying coroutine timed out
                    continue
                for sample in samples:
                    await self.add_sample_to_batches(sample)
            debug_file.write(f"Experience queue length in _create_batches after processing: {len(self.experience_queue)}\n")
            debug_file.flush()
            self.experience_queue = []
        
            await self.dispatch_batches()
            debug_file.write("Last batch dispatched\n")
            debug_file.flush()
            await self.reset_batches()
            await self.dispatch_sentinel()
            debug_file.write("Sentinel dispatched\n")
            debug_file.flush()

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
    ray.get(batcher.register_training_process.remote(0, 25000))
    ray.get(batcher.generate_experience.remote([{'input_token_ids': [100264, 882, 100266, 4438, 1053, 499, 12849, 279, 8286, 315,
            264, 6211, 1903, 315, 279, 11552, 315, 1403, 66818, 315,
            279, 1890, 10801, 1405, 279, 19169, 72359, 449, 279, 7479,
            315, 279, 1023, 26436, 30, 100265, 100264, 78191, 100266]}], 4))
    ray.get(batcher.start_consuming_experience.remote())
    print(ray.get(batcher.get_batch.remote(0)))



