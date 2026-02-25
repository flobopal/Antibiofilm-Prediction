from script.utils.metrics import evaluate
import torch

def test_r2():
    y_real = torch.rand(200)
    assert evaluate('r2', y_real, y_real) == 1.
    assert evaluate('r2', y_real, -y_real) < 0
    assert 0.7 < evaluate('r2', y_real, y_real-0.1*torch.rand(200)) < 1
    assert evaluate('r2', y_real, torch.rand(200)) < 0