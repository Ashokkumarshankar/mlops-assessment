import json
import os
import random

import numpy as np
import torch
from models.config import mlflow
from typing import Any, Dict, List

def seed_everything(seed=73):
    """Set seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def load_dict(path: str):
    """
    Load a dictionary from a JSON's filepath.

    """
    with open(path) as fp:
        d = json.load(fp)
    return d

def save_dict(d: Dict, path: str, cls: Any = None, sortkeys: bool = False) -> None:
    """
    Save a dictionary to a specific location.

    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, "w") as fp:
        json.dump(d, indent=2, fp=fp, cls=cls, sort_keys=sortkeys)
        fp.write("\n")


def get_run_id(experiment_name: str, trial_id: str) -> str:
    """Get the MLflow run ID for a specific Ray trial ID.

    """
    trial_name = f"TorchTrainer_{trial_id}"
    run = mlflow.search_runs(experiment_names=[experiment_name], filter_string=f"tags.trial_name = '{trial_name}'").iloc[0]
    return run.run_id