import argparse
import asyncio
from datetime import timedelta
import gc
import os
import time
from typing import List
import uuid
import torch
import ray
from setup_model import setup_model  # Updated function
from vllm_registry import get_or_create_registry
from sample_processing_utils import get_output_logits_indices, get_input_for_logprobs
import logging
# Set logging level to debug
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.debug("Logging initialized at DEBUG level in logprob_worker.py")


def init_logprob_dist_env():
    # torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    # args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=180))
    # args.global_rank = torch.distributed.get_rank()
    tensor = torch.ByteTensor([False]).cuda()
    torch.distributed.all_reduce(tensor)
    torch.distributed.barrier()

@ray.remote(num_gpus=1, num_cpus=4)
class LogprobWorker:
    def __init__(self, model_path: str, worker_id: str, max_tokens_per_gpu: int = 23000):
        self.args = argparse.Namespace(
            model_name_or_path=model_path,
            worker_id=worker_id,
            fsdp_sharding_strategy="FULL_SHARD",
            loss_chunksize=None,
        )
        self.worker_id = worker_id
        self.max_tokens_per_gpu = max_tokens_per_gpu
        print(f"Initializing {self.__class__.__name__} for logprobs with model path: {model_path}")
        init_logprob_dist_env()
        self.model = setup_model(self.args).cuda()
        self.device = next(self.model.parameters()).device
        self.registry = get_or_create_registry("logprob_vllm_registry")
        self.batching_queue = asyncio.Queue()
        self.setup_registration()
        self._centralizer_loop = asyncio.create_task(self._centralize_inference_requests())
    
    def setup_registration(self):
        try:
            ray.get(self.registry.register.remote(service_id=self.worker_id))
            print(f"Worker {self.worker_id} registered.")
        except Exception as e:
            print(f"Error during registration for worker {self.worker_id}: {e}")

    def free_memory(self):
        gc.collect()
        torch.cuda.empty_cache()
    
    async def update_weights(self, new_state_dict: dict):
        '''kill the centralizer loop before updating the weights'''
        await self.batching_queue.put(None)
        await self._centralizer_loop
        
        self.free_memory()
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data = new_state_dict[name].to(param.device).to(param.dtype)
        print(f"Model loaded successfully on service {self.worker_id}.")
        # self.model = setup_model(self.args, model).cuda()
        self._centralizer_loop = asyncio.create_task(self._centralize_inference_requests())
        print(f"Logprob weights updated successfully on service {self.worker_id}.")
        return True

    async def inference(self, sample: dict, **kwargs) -> dict:
        """
        Compute log probabilities synchronously using get_per_token_logps, 
        but provide an asynchronous interface.
        Expected sample dict includes 'sample_ids'
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self.batching_queue.put_nowait((future, sample))
        result = await future
        return result
    
    async def _centralize_inference_requests(self):
        # current_request acts as a pending request that didn't fit in the previous batch.
        current_request = None  
        while True:
            # If there's no pending request, get one from the queue.
            # Otherwise, use the one left from the previous batch.
            current_request = await self.batching_queue.get() if current_request is None else current_request
            if current_request is None:
                return None
            logging.debug(f"\033[1;38;2;255;165;0m _centralize_inference_requests line 118: \033[0m got request with length {len(current_request[1]['sample_ids'])}")
            logging.debug(f"\033[1;38;2;255;165;0m _centralize_inference_requests line 118: \033[0m length of batching queue: {self.batching_queue.qsize()}")
            inference_requests = [current_request]
            total_length = len(current_request[1]['sample_ids'])
            while True:
                try:
                    # Attempt to retrieve the next request.
                    current_request = await asyncio.wait_for(self.batching_queue.get(), timeout=1)
                    if current_request is None:
                        logging.debug(f"\033[1;38;2;255;255;0m _centralize_inference_requests line 142: \033[0m received sentinel")
                        return None
                    logging.debug(f"\033[1;38;2;255;165;0m _centralize_inference_requests line 129: \033[0m got request with length {len(current_request[1]['sample_ids'])}")
                    logging.debug(f"\033[1;38;2;255;165;0m _centralize_inference_requests line 129: \033[0m length of batching queue: {self.batching_queue.qsize()}")
                    len_request = len(current_request[1]['sample_ids'])
                    # If adding this request would exceed the maximum, break and keep it for next round.
                    if total_length + len_request > self.max_tokens_per_gpu:
                        logging.debug(f"\033[1;38;2;255;20;147m _centralize_inference_requests line 147: \033[0m adding this request would exceed the maximum, breaking, total_length: {total_length}, len_request: {len_request}, max_tokens_per_gpu: {self.max_tokens_per_gpu}")
                        break
                    # Otherwise, include it in the current batch.
                    total_length += len_request
                    inference_requests.append(current_request)
                except asyncio.TimeoutError:
                    # Timeout: no more requests available now.
                    logging.debug(f"\033[1;38;2;255;0;255m _centralize_inference_requests line 153: \033[0m no more items in the batching queue timeout")
                    current_request = None
                    break
            if inference_requests:
                futures, samples = zip(*inference_requests)
                samples_with_logprobs = self._compute_logprobs(samples)
                logging.debug(f"\033[1;38;2;0;255;255m _centralize_inference_requests line 160: \033[0m computed samples_with_logprobs length of batch_queue: {self.batching_queue.qsize()}")
                for future, sample_with_logprobs in zip(futures, samples_with_logprobs):
                    if not future.done():
                        future.set_result(sample_with_logprobs)

    def _compute_logprobs(self, samples: List[dict]) -> dict:
        """
        Synchronously compute log probabilities from the input sample.
        """
        output_indices, _ = get_output_logits_indices(samples, self.device)
        input_ids, position_ids, labels = get_input_for_logprobs(samples, output_indices, self.device)

        self.model.eval()
        with torch.no_grad():
            log_probs = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                labels=labels,
            ).loss

        sample_lens = [len(s['sample_ids']) for s in samples]
        log_probs = torch.split(log_probs, sample_lens)
        for s, log_prob in zip(samples, log_probs):
            s['sample_logprobs'] = log_prob.tolist()
        return samples
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start logprob worker service")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model weights")
    args = parser.parse_args()
    import logging
    logging.basicConfig(level=logging.DEBUG)
    ray.init(address="auto", namespace="test")
    print(ray.get_runtime_context().get_job_id())
    service_id = f"logprob_worker_{str(uuid.uuid4())}"
    init_logprob_dist_env()
    # worker = LogprobWorker.options(
    #     name=service_id,
    #     num_gpus=1,
    #     num_cpus=4,
    #     runtime_env={
    #         "env_vars": dict(os.environ)
    #     },
    #     # runtime_env={
    #     #     "pip": [f"-r {os.path.dirname(os.path.abspath(__file__))}/requirements_fsdp.txt"]
    #     # },
    # ).remote(args.model_path, service_id)
    from IPython import embed
    embed()

    while True:
        try:
            ray.get_actor("generation_vllm_registry")
            ray.get_actor("logprob_vllm_registry")
            print("logprob_vllm_registry found.")
            break
        except Exception:
            print("logprob_vllm_registry not found, sleeping for 2 seconds...")
            time.sleep(2)
    
    from transformers import AutoModelForCausalLM
    model_ = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    )
    new_state_dict = model_.state_dict()
    from accelerate import init_empty_weights
    with init_empty_weights():  
        model = AutoModelForCausalLM.from_pretrained(
            # args.model_path,
            "/dev/shm/phi-4",
        )
    model.load_state_dict(new_state_dict, assign=True)
    worker_id = "logprob_worker_1"
    args_ = argparse.Namespace(
            model_name_or_path=args.model_path,
            worker_id=worker_id,
            fsdp_sharding_strategy="FULL_SHARD",
        )
    model, accelerator = setup_model(args_, model)
    input_ids = [
            100264, 882, 100266, 4438, 1053, 499, 12849, 279, 8286, 315,
            264, 6211, 1903, 315, 279, 11552, 315, 1403, 66818, 315,
            279, 1890, 10801, 1405, 279, 19169, 72359, 449, 279, 7479,
            315, 279, 1023, 26436, 30, 100265, 100264, 78191, 100266
        ]
    batch_ids = torch.tensor([input_ids]).to(accelerator.device)
    position_ids = torch.arange(len(input_ids)).unsqueeze(0).to(accelerator.device)
    output_indices = (position_ids[:,1:] - 1).squeeze(0)
    model_, accelerator_ = setup_model(args_, model)
    with torch.no_grad():
        log_probs = get_per_token_logps(model, batch_ids, position_ids, output_indices)
        log_probs_ = get_per_token_logps(model_, batch_ids, position_ids, output_indices)
    print(log_probs_ - log_probs)
    ray.get(worker.update_weights.remote(new_state_dict))
    # for p_,p in zip(model.parameters(), model_.parameters()):
    #     if not torch.allclose(p_, p):
    #         print(f"Parameters do not match: {p_.shape} != {p.shape}")
    # from IPython import embed
    # embed()
    
    async def main():
        input_ids = [
            100264, 882, 100266, 4438, 1053, 499, 12849, 279, 8286, 315,
            264, 6211, 1903, 315, 279, 11552, 315, 1403, 66818, 315,
            279, 1890, 10801, 1405, 279, 19169, 72359, 449, 279, 7479,
            315, 279, 1023, 26436, 30, 100265, 100264, 78191, 100266
        ]
        actor_registry = get_or_create_registry("generation_vllm_registry")
        samples_with_experience = await actor_registry.inference_balanced.remote(
            {'input_token_ids': input_ids},
            n=2,
            temperature=0.8,
        )
        registry = get_or_create_registry("logprob_vllm_registry")
        tasks = [registry.inference_balanced.remote(s) for s in samples_with_experience]
        samples_with_logprobs = await asyncio.gather(*tasks)
        return samples_with_logprobs
    
    results = asyncio.run(main())
    print(results)
    
    while True:
        time.sleep(10000)
