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
        self.server_failures = {}  # mapping from service_id -> failure expiration time.
        self.cooldown = 30  # seconds to wait before retrying a failing server
        self.balancer_lock = asyncio.Lock()

    def register(self, service_id: str):
        self.actors[service_id] = service_id
        return f"{service_id} registered."

    def deregister(self, service_id: str):
        if service_id in self.actors:
            del self.actors[service_id]
            return f"{service_id} deregistered."
        return f"{service_id} not found."

    def get_actors(self):
        return [ray.get_actor(name, namespace="test") for name in self.actors.values()]
    
    async def acquire_actor(self, num_requests: int) -> str:
        async with self.balancer_lock:
            now = time.time()
            healthy_servers = [s for s in self.actors.keys() if self.server_failures.get(s, 0) <= now]
            candidate_servers = healthy_servers if healthy_servers else list(self.actors.keys())
            chosen_server = min(candidate_servers, key=lambda s: self.actor_current_load[s])
            self.actor_current_load[chosen_server] += num_requests
            return chosen_server

    async def release_actor(self, service_id: str, num_requests: int, error: bool = False):
        async with self.balancer_lock:
            if error:
                self.server_failures[service_id] = time.time() + self.cooldown
            self.actor_current_load[service_id] -= num_requests

    async def inference_balanced(self, 
                                 input_data: dict, 
                                 **kwargs) -> list[dict]:
        n = kwargs.get("n", 1)
        chosen_service_id = await self.acquire_actor(n)
        actor = ray.get_actor(self.actors[chosen_service_id], namespace="test")
        try:
            result = await actor.inference.remote(input_data, **kwargs)
        except Exception as e:
            await self.release_actor(chosen_service_id, n, error=True)
            raise e
        else:
            await self.release_actor(chosen_service_id, n)
            return result


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
        


