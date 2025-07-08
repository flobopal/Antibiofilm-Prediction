from typing import Optional
import torch.optim.lr_scheduler as lr_scheduler

def get_scheduler(optimizer, scheduler_name: str, **kwargs) -> Optional[lr_scheduler.LRScheduler]:
    """
    Returns a PyTorch scheduler according to the given name and additional parameters.

    Args:
        optimizer: PyTorch optimizer to which the scheduler will be applied.
        scheduler_name (str): Name of the scheduler. Examples:
            'step', 'exponential', 'cosine', 'reduceonplateau', 'none'.
        **kwargs: Specific parameters for each scheduler.

    Returns:
        PyTorch scheduler or None if 'none' is specified.
    """
    name = scheduler_name.lower()

    if name == 'step':
        return lr_scheduler.StepLR(optimizer, **kwargs)
    elif name == 'exponential':
        return lr_scheduler.ExponentialLR(optimizer, **kwargs)
    elif name == 'cosine':
        return lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    elif name == 'reduceonplateau':
        return lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    elif name == 'none':
        return None
    else:
        raise ValueError(f"Scheduler '{scheduler_name}' no soportado.")