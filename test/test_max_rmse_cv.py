import importlib

import numpy as np
import optuna
import pandas as pd
import pytest
import torch

from model.train_and_optimize.optimizer import FullModelObjective, do_study
from script.tasks.cross_validate import cross_validate


@pytest.fixture
def dataset():
    # Unequal folds, interleaved organisms, and nonunique dataframe indices.
    return pd.DataFrame({
        'kfold': ['z', 'z', 'a', 'a', 'z'],
        'organism': ['A', 'B', 'A', 'B', 'A'],
        'y': [0., 10., 2., 10., 4.],
        'prediction': [0., 10., 0., 10., 4.],
        'row_id': [0., 1., 2., 3., 4.],
    }, index=[8, 2, 8, 0, 9])


@pytest.mark.parametrize('balance', [False, True])
@pytest.mark.parametrize('include_features', [False, True])
def test_cv_pools_predictions_before_scoring(dataset, monkeypatch, balance, include_features):
    cv_module = importlib.import_module('script.tasks.cross_validate')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    trained_models = []

    class HeldOutModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(()))

        def forward(self, x, *unused):
            assert not torch.is_grad_enabled()
            assert not self.training
            assert set(x[:, 1].tolist()).isdisjoint(self.training_ids)
            return x[:, 0] + self.bias

    def fake_train(*, model, train_loader, **kwargs):
        model.training_ids = set(train_loader.dataset.tensors[0][:, 1].tolist())
        trained_models.append(model)

    monkeypatch.setattr(cv_module, 'train_model', fake_train)
    result = cross_validate(
        dataset, lambda *dims: HeldOutModel(), features_start='prediction',
        output_column='y', organism_column='organism', metrics='max_rmse',
        optimizer_class=torch.optim.SGD, optimizer_kwargs={'lr': 0.01},
        task_type='regression', normalize_features=False, verbose=False,
        balance_organism=balance, include_organism_features=include_features,
    )
    # A: pooled MSE=4/3, population variance=8/3; B: zero error.
    # Averaging the fold maxima would instead give (0 + 20) / 2 = 10.
    assert result == pytest.approx(np.sqrt(0.5))
    assert len(trained_models) == 2


class TinyObjective(FullModelObjective):
    def suggest_params(self, trial):
        return dict(
            embed_dim=4, hidden_dims=[4], activations='relu', dropout=0.,
            lr=0.001, num_epochs=1, scheduler_name='none', scheduler_kwargs={},
            last_activation_params=None,
        )


def study_kwargs(dataset, tmp_path):
    return dict(
        dataset=dataset, database=f"sqlite:///{tmp_path / 'study.db'}",
        name='pooled', objective=TinyObjective, n_trials=1,
        features_start='prediction', output_column='y',
        organism_column='organism', fold_column='kfold', normalize_features=False,
    )


@pytest.mark.parametrize('metric,direction', [('max_rmse', 'MINIMIZE'), ('r2', 'MAXIMIZE')])
def test_do_study_trains_and_scores(dataset, tmp_path, monkeypatch, metric, direction):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    torch.manual_seed(42)
    study = do_study(**study_kwargs(dataset, tmp_path), metric=metric)
    assert study.direction.name == direction
    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
    assert np.isfinite(study.best_value)


def test_do_study_rejects_existing_opposite_direction(dataset, tmp_path):
    kwargs = study_kwargs(dataset, tmp_path)
    study = optuna.create_study(
        storage=kwargs['database'], study_name=kwargs['name'], direction='maximize',
    )
    with pytest.raises(ValueError, match='Use a new study name'):
        do_study(**kwargs, metric='max_rmse')
    assert study.trials == []


@pytest.mark.parametrize('invalid', ['missing_organism', 'missing_fold', 'one_fold'])
def test_cv_rejects_invalid_grouping_before_training(dataset, invalid):
    organism_column = 'organism'
    if invalid == 'missing_organism':
        organism_column = None
    elif invalid == 'missing_fold':
        dataset.iloc[0, dataset.columns.get_loc('kfold')] = None
    else:
        dataset['kfold'] = 'only'

    def factory(*dims):
        pytest.fail('Invalid grouping should be rejected before constructing the model')

    with pytest.raises(ValueError):
        cross_validate(
            dataset, factory, features_start='prediction', output_column='y',
            organism_column=organism_column, metrics='max_rmse',
            optimizer_class=torch.optim.SGD, optimizer_kwargs={'lr': 0.01},
            task_type='regression', normalize_features=False,
        )
