from model.interaction import MoleculeOrganismInteraction
import torch

def test_output_shape():
    batch_size = 4
    fd, fo, fk, heads = 16, 8, 5, 2

    model = MoleculeOrganismInteraction(fd=fd, fo=fo, fk=fk, num_heads=heads)
    Xd = torch.randn(batch_size, fd)
    Xp = torch.randn(batch_size, fo)

    out = model(Xd, Xp)
    assert out.shape == (batch_size, fd), f"Output shape incorrect: {out.shape}"

def test_deterministic_output():
    torch.manual_seed(42)
    batch_size = 2
    fd, fo, fk, heads = 10, 6, 4, 1

    model = MoleculeOrganismInteraction(fd=fd, fo=fo, fk=fk, num_heads=heads, dropout=0)
    model.eval()  # disable dropout

    Xd = torch.randn(batch_size, fd)
    Xp = torch.randn(batch_size, fo)

    out1 = model(Xd, Xp)
    out2 = model(Xd, Xp)

    assert torch.allclose(out1, out2), "Outputs differ between runs in eval mode"

def test_dropout_effect():
    batch_size = 2
    fd, fo, fk, heads = 10, 6, 4, 1

    model = MoleculeOrganismInteraction(fd=fd, fo=fo, fk=fk, num_heads=heads, dropout=0.5)
    Xd = torch.randn(batch_size, fd)
    Xp = torch.randn(batch_size, fo)

    model.train()
    out_train = model(Xd, Xp)
    model.eval()
    out_eval = model(Xd, Xp)

    # Dropout active in train, off in eval → outputs should differ
    assert not torch.allclose(out_train, out_eval), "Dropout not affecting outputs"

def test_forward_backward():
    batch_size = 3
    fd, fo, fk, heads = 12, 7, 5, 1

    model = MoleculeOrganismInteraction(fd=fd, fo=fo, fk=fk, num_heads=heads)
    Xd = torch.randn(batch_size, fd, requires_grad=True)
    Xp = torch.randn(batch_size, fo, requires_grad=True)

    out = model(Xd, Xp).sum()
    out.backward()

    assert Xd.grad is not None, "No gradient for Xd"
    assert Xp.grad is not None, "No gradient for Xp"