from collections import defaultdict
from typing import Union
import ray
import asyncio
import time


@ray.remote
class VLLMRegistry:
    def __init__(self):
        # Mapping from unique service IDs to actor names.
        self.actors = {}
        self.actor_current_load = defaultdict(int)
        self.balancer_lock = asyncio.Condition(asyncio.Lock())
        self.last_weights = ray.put(None)

    async def register(self, service_id: str):
        async with self.balancer_lock:
            self.actors[service_id] = service_id
            self.balancer_lock.notify_all()
            print("\033[1;38;2;57;255;20mActor registered! #########################################\033[0m")
            return f"{service_id} registered."

    async def deregister(self, service_id: str):
        async with self.balancer_lock:
            self.actors.pop(service_id, None)
            self.actor_current_load.pop(service_id, None)
            print("\033[1;38;2;57;255;20mActor deregistered! #########################################\033[0m")
            return f"{service_id} deregistered."
    
    def get_last_weights(self):
        return self.last_weights
    
    def update_weights(self, new_state_dict: dict):
        self.last_weights = ray.put(new_state_dict)

    def get_actors(self):
        return [ray.get_actor(name, namespace="test") for name in self.actors.values()]
    
    async def acquire_actor(self, num_requests: int) -> str:
        async with self.balancer_lock:
            # Wait until there is an actor available
            while len(self.actors) == 0:
                print("\033[1;38;2;57;255;20mWaiting for actor to be available #########################################\033[0m")
                await self.balancer_lock.wait()
                print("\033[1;38;2;57;255;20mActor available!!!!! #########################################\033[0m")
            chosen_server = min(self.actors.keys(), key=lambda s: self.actor_current_load[s])
            self.actor_current_load[chosen_server] += num_requests
            return chosen_server

    async def release_actor(self, service_id: str, num_requests: int, error: bool = False):
        async with self.balancer_lock:
            self.actor_current_load[service_id] -= num_requests

    async def inference_balanced(self, 
                                 input_data: dict, 
                                 **kwargs) -> list[dict]:
        n = kwargs.get("n", 1)
        for _ in range(3):
            chosen_service_id = await self.acquire_actor(n)
            try:
                actor = ray.get_actor(self.actors[chosen_service_id], namespace="test")
                result = await actor.inference.remote(input_data, **kwargs)
                await self.release_actor(chosen_service_id, n)
                return result
            except Exception as e:
                await self.deregister(chosen_service_id)
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
        


