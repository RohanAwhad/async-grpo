'''
The purpose of this dispatcher is to handle conflicting environments between training and vllm environments. 
Ray has an environment management that onle installs pip packages on top of the base environment. 
So having different requirements.txt was the only way to figure this out.
'''

import argparse
import os
import time
import uuid
import ray

@ray.remote(num_cpus=1)
class WorkerFactory:
    def create_worker(self, mode: str, model_path: str, tensor_parallel_size: int, max_num_seqs: int, max_tokens_per_gpu: int = 23000):
        """
        Instantiate the appropriate worker on the remote process after the runtime environment
        is set up. This defers the import of worker-specific modules to the worker process.
        """
        if mode == "generation":
            from vllm_worker import GenerationVLLMWorker  # lazy import on remote worker
            service_id = f"generation_worker_{uuid.uuid4()}"
            worker = GenerationVLLMWorker.options(
                name=service_id,
                num_gpus=tensor_parallel_size,
                num_cpus=4,
            ).remote(
                model_path=model_path,
                worker_id=service_id,
                tensor_parallel_size=tensor_parallel_size,
                max_num_seqs=max_num_seqs,
            )
        elif mode == "logprob":
            from logprob_worker import LogprobWorker  # lazy import on remote worker
            service_id = f"logprob_worker_{uuid.uuid4()}"
            worker = LogprobWorker.options(
                name=service_id,
                num_gpus=1,
                num_cpus=4,
            ).remote(
                model_path=model_path,
                worker_id=service_id,
                max_tokens_per_gpu=max_tokens_per_gpu,
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")
        return worker

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Worker Dispatcher for Generation and Logprob Workers"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model weights")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of tensor parallel units")
    parser.add_argument("--max_num_seqs", type=int, default=16,
                        help="Maximum number of sequences to generate")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["generation", "logprob"],
                        help="Worker mode: generation or logprob")
    parser.add_argument("--max_tokens_per_gpu", type=int, default=23000,
                        help="Maximum tokens per GPU for logprob worker")
    args = parser.parse_args()

    # Initialize Ray.
    ray.init(address="auto", namespace="test")

    print(f"Launching {args.mode} worker ...")
    
    runtime_env = {"env_vars": dict(os.environ)}
    runtime_env["env_vars"].pop("CUDA_VISIBLE_DEVICES", None)
    if args.mode == "generation":
        runtime_env["pip"] = [f"-r {os.path.join(os.path.dirname(__file__), 'requirements_vllm.txt')}"]
    elif args.mode == "logprob":
        runtime_env["pip"] = [f"-r {os.path.join(os.path.dirname(__file__), 'requirements_fsdp.txt')}"]

    # Create the remote factory with the proper runtime_env so that its remote methods
    # execute in the customized environment.
    factory = WorkerFactory.options(runtime_env=runtime_env).remote()
    worker = ray.get(factory.create_worker.remote(
        mode=args.mode,
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.max_num_seqs,
        max_tokens_per_gpu=args.max_tokens_per_gpu,
    ))

    # Wait for the appropriate registry to be available before moving on.
    registry_name = "generation_vllm_registry" if args.mode == "generation" else "logprob_vllm_registry"
    print(f"Waiting for registry {registry_name} to be available...")
    while True:
        try:
            ray.get_actor(registry_name)
            print(f"Registry {registry_name} found.")
            break
        except Exception:
            print(f"Registry {registry_name} not available, sleeping for 5 second...")
            time.sleep(5)

    while True:
        try:
            ray.get_actor("generation_vllm_registry")
            ray.get_actor("logprob_vllm_registry")
            print("logprob_vllm_registry found.")
            break
        except Exception:
            print("logprob_vllm_registry not found, sleeping for 2 seconds...")
            time.sleep(2)
    
    

    # Keep the process alive (so that the worker remains registered).
    try:
        while True:
            time.sleep(10000)
    except KeyboardInterrupt:
        print("Worker dispatcher is shutting down.")

'''
mamba activate ray
set model /dev/shm/qwen7b-math-base
for i in (seq 0 3)
    if test $i -lt 2
        echo "Launching generation worker on GPU $i..."
        python worker_dispatcher.py \
            --model_path /dev/shm/qwen7b-math-base \
            --mode generation \
            --tensor_parallel_size 1 \
            --max_num_seqs 128 &
    else
        # CUDA_VISIBLE_DEVICES="$i" python logprob_worker.py --model_path /dev/shm/phi-4 &
        echo "Launching logprob worker on GPU $i..."
        torchrun --nproc_per_node=1 --master_port=1234$i worker_dispatcher.py \
            --model_path /dev/shm/qwen7b-math-base \
            --mode logprob &
    end
end

for i in (seq 0 3)
    echo "Launching generation worker on GPU $i..."
    python worker_dispatcher.py \
        --model_path /dev/shm/qwen7b-math-base \
        --mode generation \
        --tensor_parallel_size 1 \
        --max_num_seqs 128 &
end


for i in (seq 6 7)
    echo "Launching logprob worker on GPU $i..."
    CUDA_VISIBLE_DEVICES="$i" RAY_DISABLE_GPU_AFFINITY=1 torchrun --nproc_per_node=1 --master_port=1234$i worker_dispatcher.py \
        --model_path /dev/shm/phi-4 \
        --mode logprob &
end
'''
