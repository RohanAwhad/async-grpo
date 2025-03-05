from collections import defaultdict
from accelerate import Accelerator
import torch

class BatchMetrics:
    excluded_vars: list = []
    totals: dict = defaultdict(float)
    minibatch_metrics: dict = defaultdict(float)

    def register_global_metric(self, variable_name: str, value: float = 0.0):
        """
        Register a global metric.
        """
        setattr(self, variable_name, value)
        self.excluded_vars.append(variable_name)

    def accumulate_minibatch_metrics(self, **kwargs):
        """
        Accumulate metrics using a dictionary of new values.
        The keys of new_values should correspond to the attributes of this dataclass.
        This method adds the new value to the existing metric. 
        """
        for key, value in kwargs.items():
            self.minibatch_metrics[key] += value

    def reduce_batch_metrics(self, accelerator: Accelerator):
        """
        Reduce the minibatch metrics across all processes.
        """
        # Create a tensor from the minibatch_metrics values in the order of keys
        keys = list(self.minibatch_metrics.keys())
        tensor = torch.tensor([float(self.minibatch_metrics[k]) for k in keys], device=accelerator.device)
        reduced_tensor = accelerator.reduce(tensor, reduction="sum")
        for key, value in zip(keys, reduced_tensor.tolist()):
            self.totals[key] += value
        self.minibatch_metrics = defaultdict(float)

    def reset_batch(self):
        """Reset all metrics in totals except for the keys in excluded_vars."""
        for key in self.totals:
            if key not in self.excluded_vars:
                self.totals[key] = 0.0