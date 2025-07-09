import torch
from model.train_and_optimize.trainer import Trainer

class DummyInteractionParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class DummyFFParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class DummyModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self.linear = torch.nn.Linear(1,1)

def test_trainer_init_and_getters(monkeypatch):
    # Patch imported classes to avoid dependency issues
    monkeypatch.setattr("model.trainer.MoleculeOrganismInteractionParams", DummyInteractionParams)
    monkeypatch.setattr("model.trainer.FeedForwardNetworkParams", DummyFFParams)
    monkeypatch.setattr("model.trainer.FullModel", DummyModel)

    Xd = torch.randn(8, 5)
    Xp = torch.randn(8, 3)
    y = torch.randn(8, 1)
    trainer = Trainer(
        Xd=Xd,
        Xp=Xp,
        y=y,
        embed_dim=16,
        hidden_dims=[32, 16],
        activations='relu',
        num_heads=2,
        pooling='max',
        dropout=0.2,
        lr=0.01,
        num_epochs=10,
        scheduler_name='step',
        scheduler_kwargs={'step_size': 5},
        verbose=False
    )

    # Test get_interaction_params
    mo_params = trainer.get_interaction_params()
    assert mo_params.mol_input_dim == 5
    assert mo_params.org_input_dim == 3
    assert mo_params.embed_dim == 16
    assert mo_params.num_heads == 2
    assert mo_params.pooling == 'max'
    assert mo_params.dropout == 0.2

    # Test get_ff_params
    ff_params = trainer.get_ff_params()
    assert ff_params.input_dims == 5
    assert ff_params.hidden_dims == [32, 16]
    assert ff_params.activations == 'relu'
    assert ff_params.dropout == 0.2

    # Test get_model returns DummyModel
    model = trainer.get_model()
    assert isinstance(model, DummyModel)

    # Test get_loader returns DataLoader with correct length
    loader = trainer.get_loader()
    batch = next(iter(loader))
    assert len(batch) == 3
    assert batch[0].shape[1] == 5
    assert batch[1].shape[1] == 3
    assert batch[2].shape[1] == 1

def test_train_calls_train_model(monkeypatch):
    called = {}

    def dummy_train_model(model, train_loader, val_loader, optimizer, task_type, **kwargs):
        called['train_model'] = True
        assert model is not None
        assert train_loader is not None
        assert optimizer is not None
        assert task_type == 'regression'
        return "trained"

    monkeypatch.setattr("model.trainer.MoleculeOrganismInteractionParams", DummyInteractionParams)
    monkeypatch.setattr("model.trainer.FeedForwardNetworkParams", DummyFFParams)
    monkeypatch.setattr("model.trainer.FullModel", DummyModel)
    monkeypatch.setattr("model.trainer.FullModel", DummyModel)
    monkeypatch.setattr("model.trainer.train_model", dummy_train_model)

    Xd = torch.randn(4, 2)
    Xp = torch.randn(4, 2)
    y = torch.randn(4, 1)
    trainer = Trainer(Xd, Xp, y, 8, [8], verbose=False)
    trainer.train()
    assert called.get('train_model', False)

def test_cross_validation_calls_cross_validate_model(monkeypatch):
    called = {}

    def dummy_cross_validate_model(model_class, model_kwargs, Xd, Xp, y, optimizer_class, optimizer_kwargs, task_type, **kwargs):
        called['cross_validate_model'] = True
        assert model_class is DummyModel
        assert 'mo_params' in model_kwargs
        assert 'ff_params' in model_kwargs
        assert Xd.shape[0] == 6
        assert Xp.shape[0] == 6
        assert y.shape[0] == 6
        assert optimizer_class is not None
        assert optimizer_kwargs['lr'] == 0.01
        assert task_type == 'regression'
        return "cv_result"

    monkeypatch.setattr("model.trainer.MoleculeOrganismInteractionParams", DummyInteractionParams)
    monkeypatch.setattr("model.trainer.FeedForwardNetworkParams", DummyFFParams)
    monkeypatch.setattr("model.trainer.FullModel", DummyModel)
    monkeypatch.setattr("model.trainer.cross_validate_model", dummy_cross_validate_model)

    Xd = torch.randn(6, 2)
    Xp = torch.randn(6, 2)
    y = torch.randn(6, 1)
    trainer = Trainer(Xd, Xp, y, 8, [8], lr=0.01, verbose=False)
    result = trainer.cross_validation(kfolds=3)
    assert called.get('cross_validate_model', False)
    assert result == "cv_result"