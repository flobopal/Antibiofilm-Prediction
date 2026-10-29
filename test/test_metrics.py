from script.utils.metrics import evaluate
import torch
import pytest


def test_max_rmse_uses_worst_organism_and_population_std():
    # A: RMSE=sqrt(2), std=1; B: perfect predictions despite larger scale.
    y = torch.tensor([0., 2., 10., 30.])
    pred = torch.tensor([0., 0., 10., 30.], requires_grad=True)
    result = evaluate('max_rmse', y, pred, organisms=['A', 'A', 'B', 'B'])
    assert result == pytest.approx(2 ** 0.5)


@pytest.mark.parametrize('y,pred,expected', [
    ([2.], [2.2], 2.),
    ([2., 2.], [2.2, 2.2], 2.),
    ([0., 0.1], [0.1, 0.2], 1.),
    ([2., 2.], [2., 2.], 0.),
])
def test_max_rmse_std_floor(y, pred, expected):
    assert evaluate('max_rmse', y, pred, organisms=['A'] * len(y)) == pytest.approx(expected)


def test_max_rmse_ignores_fold_frequency_weights():
    assert evaluate(
        'max_rmse', [0., 2.], [0., 0.], organisms=['A', 'A'],
        sample_weight=torch.tensor([100., 1.]),
    ) == pytest.approx(2 ** 0.5)


@pytest.mark.parametrize('y,pred,organisms', [
    ([], [], []),
    ([0.], [0., 1.], ['A']),
    ([0.], [float('nan')], ['A']),
    ([0.], [0.], [None]),
    ([[0.]], [[0.]], ['A']),
    ([0.], [0.], None),
])
def test_max_rmse_rejects_invalid_inputs(y, pred, organisms):
    with pytest.raises(ValueError):
        evaluate('max_rmse', y, pred, organisms=organisms)


def test_weighted_r2_unchanged():
    assert evaluate(
        'r2', [0., 1., 2.], [0., 1., 1.], sample_weight=[1., 1., 2.]
    ) == pytest.approx(3 / 11)

def test_r2():
    y_real = torch.rand(200)
    assert evaluate('r2', y_real, y_real) == 1.
    assert evaluate('r2', y_real, -y_real) < 0
    assert 0.7 < evaluate('r2', y_real, y_real-0.1*torch.rand(200)) < 1
    assert evaluate('r2', y_real, torch.rand(200)) < 0
