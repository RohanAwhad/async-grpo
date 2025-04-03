from collections import defaultdict
from typing import Union
import ray
import asyncio
import time

# Simple debug logger that writes to a file named debug.log
DEBUG = False
def debug_log(message: str):
    if DEBUG:
        # Retrieve the current actor's unique ID.
        try:
            current_actor_id = ray.get_runtime_context().get_actor_id()
        except Exception as e:
            current_actor_id = "unknown"
            print(f"Error getting actor ID: {e}")
        print(f"\033[1;38;2;57;255;20m{message}\033[0m")
        with open(f"debug_{current_actor_id}.log", "a") as f:
            f.write(message + "\n")

@ray.remote
class VLLMRegistry:
    def __init__(self):
        # Mapping from unique service IDs to actor names.
        self.actors = {}
        self.actor_current_load = defaultdict(int)
        self.actor_thresholds = {}  # new: track max load for each worker
        self.balancer_lock = asyncio.Condition(asyncio.Lock())
        self.last_weights = ray.put(None)

    async def register(self, service_id: str, max_load: int = 2**31-1):
        async with self.balancer_lock:
            self.actors[service_id] = service_id
            self.actor_thresholds[service_id] = max_load  # store max load threshold
            self.actor_current_load[service_id] = 0
            self.balancer_lock.notify_all()
            return f"{service_id} registered with threshold {max_load}."

    async def deregister(self, service_id: str):
        async with self.balancer_lock:
            if service_id in self.actors:
                actor_handle = ray.get_actor(service_id, namespace="test")
                ray.kill(actor_handle, no_restart=False)
                self.actors.pop(service_id, None)
                self.actor_current_load.pop(service_id, None)
                self.actor_thresholds.pop(service_id, None)  # new: remove threshold info
            return f"{service_id} deregistered."
    
    def get_last_weights(self):
        return self.last_weights
    
    def update_weights(self, new_state_dict: dict):
        '''directly put the new state in ray object storage to decrease latency'''
        self.last_weights = ray.put(new_state_dict)

    def get_actors(self):
        return [ray.get_actor(name, namespace="test") for name in self.actors.values()]
    
    async def acquire_actor(self, num_requests: int) -> str:
        async with self.balancer_lock:
            while True:
                available_actors = [s for s in self.actors if self.actor_current_load[s] + num_requests <= self.actor_thresholds[s]]
                if available_actors:
                    chosen_server = min(available_actors, key=lambda s: self.actor_current_load[s])
                    self.actor_current_load[chosen_server] += num_requests
                    return chosen_server
                await self.balancer_lock.wait()

    async def release_actor(self, service_id: str, num_requests: int, error: bool = False):
        async with self.balancer_lock:
            self.actor_current_load[service_id] -= num_requests
            self.balancer_lock.notify_all()

    async def inference_balanced(self, 
                                 sample: dict, 
                                 **kwargs) -> list[dict]:
        n = kwargs.get("n", 1)
        while True:
            chosen_service_id = await self.acquire_actor(n)
            try:
                actor = ray.get_actor(self.actors[chosen_service_id], namespace="test")
                result = await actor.inference.remote(sample=sample, **kwargs)
                await self.release_actor(chosen_service_id, n)
                return result
            except ray.exceptions.ActorUnavailableError:
                '''Doing this to avoid restarting the failing actor immediately after it's registered
                   Ray needs some time to make the actor available'''
                await self.release_actor(chosen_service_id, n)
                await asyncio.sleep(1)
            except Exception as e:
                import traceback
                debug_log("Exception during inference_balanced: " + traceback.format_exc())
                await self.deregister(chosen_service_id)
                await asyncio.sleep(1)
                # raise e
        raise Exception("All actors failed.")


# Helper function to get or create a registry actor.
def get_or_create_registry(registry_actor_name: str) -> VLLMRegistry:
    """
    Retrieve the registry actor by its name. If it does not exist,
    create a new VLLMRegistry actor with the given name.
    """
    try:
        return VLLMRegistry.options(name=registry_actor_name).remote()
    except Exception:
        return ray.get_actor(registry_actor_name)
        


