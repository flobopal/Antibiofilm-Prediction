import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from script.tasks.cross_validate import cross_validate_model
import pytest


# Simple dummy model
class DummyModel(nn.Module):
    def __init__(self, input_size_d, input_size_p, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_size_d + input_size_p, 1)

    def forward(self, Xd, Xp):
        x = torch.cat([Xd, Xp], dim=1)
        return self.linear(x).squeeze(-1)


def generate_dummy_data(n_samples=100, d_dim=10, p_dim=15):
    Xd = torch.randn(n_samples, d_dim)
    Xp = torch.randn(n_samples, p_dim)
    y = torch.randn(n_samples)
    return Xd, Xp, y


def test_cross_validate_regression_runs():
    Xd, Xp, y = generate_dummy_data()

    losses = cross_validate_model(
        model_class=DummyModel,
        model_kwargs={"input_size_d": 10, "input_size_p": 15, "hidden_size": 16},
        Xd=Xd,
        Xp=Xp,
        y=y,
        task_type="regression",
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={"lr": 0.01},
        num_epochs=3,
        batch_size=16,
        k_folds=3
    )

    assert isinstance(losses, list)
    assert len(losses) == 3
    for loss in losses:
        assert isinstance(loss, float)
        assert loss >= 0.0


def test_cross_validate_invalid_task_type():
    Xd, Xp, y = generate_dummy_data()
    with pytest.raises(ValueError):
        cross_validate_model(
            model_class=DummyModel,
            model_kwargs={"input_size_d": 10, "input_size_p": 15, "hidden_size": 16},
            Xd=Xd,
            Xp=Xp,
            y=y,
            task_type="unsupported",
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs={"lr": 0.01},
        )