import pytest
import torch
from model.decoder import FeedForwardNetwork, FeedForwardNetworkParams
from script.utils.activation_functions import get_activation

def test_feedforward_output_shape():
    batch_size = 10
    input_dim = 16
    hidden_dims = [32, 16, 8]
    activations = ['relu', 'relu', 'tanh', 'tanh']
    dropout = 0.0

    model = FeedForwardNetwork(input_dim, hidden_dims, activations, dropout)
    x = torch.randn(batch_size, input_dim)
    out = model(x)

    # Output shape debe ser (batch_size, 1)
    assert out.shape == (batch_size,)

def test_feedforward_forward_runs_without_error():
    model = FeedForwardNetwork(8, [16, 8], ['relu', 'relu', 'tanh'], 0.1)
    x = torch.randn(5, 8)
    try:
        out = model(x)
    except Exception as e:
        pytest.fail(f"Forward pass failed with exception {e}")

def test_activation_invalid_name():
    with pytest.raises(ValueError):
        get_activation('nonexistent_activation')

def test_dropout_effect():
    # Comprueba que en modo train el dropout está activo y en eval no
    model = FeedForwardNetwork(4, [8, 4], ['relu', 'tanh', 'tanh'], dropout=0.5)
    model.train()
    x = torch.ones(2, 4)
    out_train = model(x)

    model.eval()
    out_eval = model(x)

    # En eval dropout está desactivado, salida debería ser diferente
    assert not torch.equal(out_train, out_eval)

def test_params_init():
    params = FeedForwardNetworkParams(16, [8, 4], 'relu', 0.5)
    model = FeedForwardNetwork.from_params(params)
    assert len(model.model) == 9, "Not all layers have been added"
    assert model.model[0].__class__.__name__ == 'ReLU'
    assert model.model[1].p == 0.5
    assert model.model[2].in_features == 16
    assert model.model[2].out_features == 8
    assert model.model[3].__class__.__name__ == 'ReLU'
    assert model.model[4].p == 0.5
    assert model.model[5].in_features == 8
    assert model.model[5].out_features == 4
    assert model.model[6].__class__.__name__ == 'ReLU'
    assert model.model[7].p == 0.5
    assert model.model[8].in_features == 4
    assert model.model[8].out_features == 1
