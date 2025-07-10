from functools import wraps
from typing import Callable
from sklearn.metrics import r2_score
import torch
import numpy

metrics_dict = {
    'r2': r2_score
}

class Float_with_item_method(float):
    def item(self):
        return self

def list_metrics():
    return list(metrics_dict.keys())

def evaluate(metric: str, *args: numpy.ndarray | torch.Tensor) -> float:
    if metric not in metrics_dict:
        raise ValueError("allowed metrics are " + ', '.join(list_metrics()))
    args = [arg.to_cpu() if isinstance(args, torch.Tensor) else arg for arg in args]
