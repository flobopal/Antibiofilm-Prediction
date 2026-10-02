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

def evaluate(
    metric: str,
    *args: numpy.ndarray | torch.Tensor,
    sample_weight: numpy.ndarray | torch.Tensor | None = None,
) -> float:
    if metric not in metrics_dict:
        raise ValueError(f"Metric '{metric}' is not supported. Supported metrics are: {list_metrics()}")
    metric_func = metrics_dict[metric]
    args = [to_cpu(arg) for arg in args]
    kwargs = {}
    if sample_weight is not None:
        kwargs['sample_weight'] = to_cpu(sample_weight)
    return metric_func(*args, **kwargs)
