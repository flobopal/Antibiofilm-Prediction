import numpy as np
import pandas as pd
import pytest
import torch

from model.decoder import FeedForwardNetwork, FeedForwardNetworkParams, train_mlp
from model.full_model import FullModel, train_model
from model.interaction import MoleculeOrganismInteractionParams
from script.tasks.cross_validate import cross_validate


@pytest.fixture(params=["cpu", "cuda"])
def checked_optimizers(request, monkeypatch):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if request.param == "cpu":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    instances = []

    class CheckedAdam(torch.optim.Adam):
        def __init__(self, params, **kwargs):
            params = list(params)
            assert all(p.device.type == request.param for p in params)
            super().__init__(params, **kwargs)
            instances.append(self)

        def step(self, closure=None):
            params = [p for group in self.param_groups for p in group["params"]]
            # Stale optimizer parameters would receive no gradients.
            assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in params)
            before = [p.detach().clone() for p in params]
            result = super().step(closure)
            assert any(not torch.equal(old, p) for old, p in zip(before, params))
            return result

    monkeypatch.setattr(torch.optim, "Adam", CheckedAdam)
    # Force device conversions to replace Parameter objects even on CPU,
    # reproducing stale optimizer references without requiring CUDA hardware.
    previous = torch.__future__.get_overwrite_module_params_on_conversion()
    torch.__future__.set_overwrite_module_params_on_conversion(True)
    try:
        yield instances
    finally:
        torch.__future__.set_overwrite_module_params_on_conversion(previous)


@pytest.fixture
def dataset():
    return pd.DataFrame({
        "organism": ["A", "B"] * 12,
        "y": np.linspace(4., 8., 24),
        "kfold": np.repeat([0, 1, 2], 8),
        "f0": np.linspace(-1., 1., 24),
        "f1": np.sin(np.arange(24)),
    })


@pytest.mark.parametrize("trainer", [train_model, train_mlp])
def test_training_updates_the_model_parameters(trainer, dataset, tmp_path, checked_optimizers):
    torch.manual_seed(42)
    model = trainer(
        data_path=dataset,
        features_start=3,
        organism_column="organism",
        organism_encoder_path=tmp_path / "encoder.pkl",
        output_column="y",
        hidden_dims=[4],
        activations=["relu", "relu"],
        dropout=0.,
        lr=0.01,
        num_epochs=2,
        batch_size=8,
        weighted_loss=True,
    )
    assert len(checked_optimizers) == 1
    optimizer_params = checked_optimizers[0].param_groups[0]["params"]
    assert {id(p) for p in model.parameters()} == {id(p) for p in optimizer_params}


@pytest.mark.parametrize("include_organisms", [True, False])
def test_cv_updates_each_fold_model(dataset, checked_optimizers, include_organisms):
    torch.manual_seed(42)
    models = []

    def factory(fd, fo):
        if include_organisms:
            model = FullModel(
                MoleculeOrganismInteractionParams(fd, fo, 4, 0.),
                FeedForwardNetworkParams(4, [4], "relu", 0.),
            )
        else:
            model = FeedForwardNetwork(fd, [4], "relu", dropout=0.)
        models.append(model)
        return model

    score = cross_validate(
        dataset, factory, features_start=3, output_column="y",
        optimizer_class=torch.optim.Adam, optimizer_kwargs={"lr": 0.01},
        task_type="regression", organism_column="organism",
        include_organism_features=include_organisms, balance_organism=True,
        metrics="max_rmse", num_epochs=1, batch_size=8, verbose=False,
    )
    assert np.isfinite(score)
    assert len(models) == len(checked_optimizers) == 3
    for model, optimizer in zip(models, checked_optimizers):
        optimizer_params = optimizer.param_groups[0]["params"]
        assert {id(p) for p in model.parameters()} == {id(p) for p in optimizer_params}
