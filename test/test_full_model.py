import torch
from torch import nn
from model.full_model import FullModel
from model.interaction import MoleculeOrganismInteractionParams
from model.decoder import FeedForwardNetworkParams

def make_model(fd=8, fo=6, fk=4, heads=2, ff_hidden=[16]):
    mo_params = MoleculeOrganismInteractionParams(
        mol_input_dim=fd,
        org_input_dim=fo,
        embed_dim=fk,
        num_heads=heads,
        pooling='linear',
        dropout=0.0
    )
    ff_params = FeedForwardNetworkParams(
        input_dims=fd,
        hidden_dims=ff_hidden,
        dropout=0.0,
        activations='relu'
    )
    return FullModel(mo_params, ff_params)

def test_fullmodel_output_shape():
    batch_size = 5
    fd, fo = 8, 6
    model = make_model(fd=fd, fo=fo)
    Xd = torch.randn(batch_size, fd)
    Xp = torch.randn(batch_size, fo)
    out = model(Xd, Xp)
    assert out.shape == (batch_size, 1), f"Expected output shape (batch, 3), got {out.shape}"

def test_fullmodel_forward_backward():
    batch_size = 4
    fd, fo = 10, 7
    model = make_model(fd=fd, fo=fo)
    Xd = torch.randn(batch_size, fd, requires_grad=True)
    Xp = torch.randn(batch_size, fo, requires_grad=True)
    out = model(Xd, Xp).sum()
    out.backward()
    assert Xd.grad is not None, "No gradient for Xd"
    assert Xp.grad is not None, "No gradient for Xp"

def test_fullmodel_deterministic_eval():
    torch.manual_seed(123)
    model = make_model()
    model.eval()
    Xd = torch.randn(2, 8)
    Xp = torch.randn(2, 6)
    out1 = model(Xd, Xp)
    out2 = model(Xd, Xp)
    assert torch.allclose(out1, out2), "Outputs differ in eval mode"

def test_fullmodel_dropout_effect():
    mo_params = MoleculeOrganismInteractionParams(
        mol_input_dim=8,
        org_input_dim=6,
        embed_dim=4,
        num_heads=2, pooling='linear', dropout=0.5
    )
    ff_params = FeedForwardNetworkParams(
        input_dims=8,
        hidden_dims=[16, 16],
        dropout=0.5,
        activations='relu'
    )
    model = FullModel(mo_params, ff_params)
    Xd = torch.randn(3, 8)
    Xp = torch.randn(3, 6)
    model.train()
    out_train = model(Xd, Xp)
    model.eval()
    out_eval = model(Xd, Xp)
    assert not torch.allclose(out_train, out_eval), "Dropout not affecting outputs"

def test_fullmodel_from_params_consistency():
    mo_params = MoleculeOrganismInteractionParams(
        mol_input_dim=5,
        org_input_dim=4,
        embed_dim=3,
        num_heads=1, pooling='linear', dropout=0.1
    )
    ff_params = FeedForwardNetworkParams(
        input_dims=5,
        hidden_dims=[8],
        dropout=0.1,
        activations='relu'
    )
    model = FullModel(mo_params, ff_params)
    assert isinstance(model.mo, nn.Module)
    assert isinstance(model.ff, nn.Module)