import torch.optim as optim
import torch
import pytest
from script.utils.scheduler import get_scheduler

def test_get_scheduler_returns_correct_scheduler():
    model_params = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]

    optimizer = optim.Adam(model_params, lr=0.001)

    # Test regression
    scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler_name='step',
        step_size=10,
        gamma=0.5
    )
    assert scheduler is not None
    assert scheduler.__class__.__name__ == 'StepLR'

    # Test classification with cosine annealing
    scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler_name='cosine',
        T_max=5
    )
    assert scheduler is not None
    assert scheduler.__class__.__name__ == 'CosineAnnealingLR'

    # Test with none type -> should return None
    scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler_name='none'
    )
    assert scheduler is None 

    # Test with unknown type -> should raise ValueError
    with pytest.raises(ValueError):
        get_scheduler(
            optimizer=optimizer,
            scheduler_name='unknown'
        )

def test_get_scheduler_defaults():
    model_params = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]

    optimizer = optim.Adam(model_params, lr=0.001)

    # Without passing specific parameters, should return a StepLR with default values
    scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler_name='step',
        step_size=10
    )
    assert scheduler.gamma == 0.1