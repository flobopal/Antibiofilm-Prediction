from sklearn.metrics import r2_score
import torch
import numpy
import pandas as pd


def max_rmse(y_true, y_pred, organisms) -> float:
    """Worst organism RMSE / max(std(y_true, ddof=0), 0.1).

    Supply pooled out-of-fold predictions. Each observation has equal weight
    within its organism; organism-frequency training weights are not used.
    """
    y_true = numpy.asarray(to_cpu(y_true), dtype=numpy.float64)
    y_pred = numpy.asarray(to_cpu(y_pred), dtype=numpy.float64)
    organisms = numpy.asarray(to_cpu(organisms))
    if any(array.ndim != 1 for array in (y_true, y_pred, organisms)):
        raise ValueError("max_rmse expects one-dimensional targets, predictions and organisms")
    if not (len(y_true) == len(y_pred) == len(organisms)) or len(y_true) == 0:
        raise ValueError("max_rmse requires nonempty arrays of equal length")
    if not numpy.isfinite(y_true).all() or not numpy.isfinite(y_pred).all():
        raise ValueError("max_rmse requires finite targets and predictions")
    if pd.isna(organisms).any():
        raise ValueError("max_rmse requires nonmissing organism labels")

    codes, labels = pd.factorize(organisms)
    errors = []
    for code in range(len(labels)):
        mask = codes == code
        rmse = numpy.sqrt(numpy.mean((y_true[mask] - y_pred[mask]) ** 2))
        scale = max(numpy.std(y_true[mask], ddof=0), 0.1)
        errors.append(rmse / scale)
    return float(max(errors))

metrics_dict = {
    'r2': r2_score,
    'max_rmse': max_rmse,
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
    organisms: numpy.ndarray | torch.Tensor | None = None,
) -> float:
    if metric not in metrics_dict:
        raise ValueError(f"Metric '{metric}' is not supported. Supported metrics are: {list_metrics()}")
    metric_func = metrics_dict[metric]
    args = [to_cpu(arg) for arg in args]
    if metric == "max_rmse":
        if organisms is None:
            raise ValueError("organisms must be provided for max_rmse")
        # Frequency weights from separate folds must not reweight pooled groups.
        return metric_func(*args, organisms=to_cpu(organisms))
    kwargs = {}
    if sample_weight is not None:
        kwargs['sample_weight'] = to_cpu(sample_weight)
    return metric_func(*args, **kwargs)
