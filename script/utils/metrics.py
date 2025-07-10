from functools import wraps
from typing import Callable
from sklearn.metrics import r2_score
import torch
import numpy

metrics_dict = {
    'r2': r2_score
}

def list_metrics():
    return list(metrics_dict.keys())

def to_cpu(array: numpy.ndarray | torch.Tensor) -> numpy.ndarray:
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return array

def evaluate(metric: str, *args: numpy.ndarray | torch.Tensor) -> float:
    return metrics_dict.get(metric)(*map(to_cpu, args))
