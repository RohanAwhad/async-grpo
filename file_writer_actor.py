# SPDX-License-Identifier: Apache-2.0

import os
import json
import numpy as np
import torch
import ray
from datetime import datetime

@ray.remote
class FileWriterActor:
    """
    Ray actor that synchronously appends JSON records
    Supports timestamping and optional console printing.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        # Ensure directory exists
        dirpath = os.path.dirname(self.file_path)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)

    def append(self, record: dict, timestamp: bool = False, print_console: bool = False):
        """
        Append a JSON record to the file.
        :param record: dict to serialize
        :param timestamp: if True, add a 'timestamp' key
        :param print_console: if True, also print the record
        """
        if timestamp:
            record['timestamp'] = datetime.now().isoformat()
        # Sanitize scalar numpy and torch tensor values
        for k, v in record.items():
            if isinstance(v, torch.Tensor):
                record[k] = v.item()
            elif isinstance(v, np.generic):
                record[k] = v.item()
        line = json.dumps(record, separators=(',', ':'), ensure_ascii=False) + '\n'
        # Actor execution is single-threaded per actor, so no explicit lock needed
        with open(self.file_path, 'a') as f:
            f.write(line)
        # Optional console output
        if print_console:
            print(f"\033[92m{json.dumps(record, indent=4)}\033[0m")


def get_or_create_filewriter(actor_name: str, file_path: str):
    """
    Get a named FileWriterActor or create it if it doesn't exist.
    Uses the 'test' namespace for Ray actors.
    """
    try:
        return FileWriterActor.options(name=actor_name, namespace="test").remote(file_path)
    except Exception:
        return ray.get_actor(actor_name, namespace="test") 