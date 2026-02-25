import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pytest
from script.tasks import train

def test_move_batch_to_device_moves_tensors_correctly():
    Xd = torch.randn(2, 5)
    Xp = torch.randn(2, 3)
    y = torch.tensor([1.0, 0.0])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Xd_new, Xp_new, y_new = train.move_batch_to_device(Xd, Xp, y, device)

    assert Xd_new.device == device
    assert Xp_new.device == device


def test_get_criterion_regression():
    loss_fn = train.get_criterion("regression")
    assert isinstance(loss_fn, nn.MSELoss)

def test_get_criterion_binary_bce():
    loss_fn = train.get_criterion("binary", use_logits=False)
    assert isinstance(loss_fn, nn.BCELoss)

def test_get_criterion_binary_bce_logits():
    loss_fn = train.get_criterion("binary", use_logits=True)
    assert isinstance(loss_fn, nn.BCEWithLogitsLoss)

def test_get_criterion_multiclass():
    loss_fn = train.get_criterion("multiclass")
    assert isinstance(loss_fn, nn.CrossEntropyLoss)

def test_get_criterion_invalid():
    with pytest.raises(ValueError):
        train.get_criterion("unsupported")

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, Xd, Xp):
        x = torch.cat([Xd, Xp], dim=1)
        return self.linear(x)

def create_dataloader():
    Xd = torch.randn(10, 2)
    Xp = torch.randn(10, 2)
    y = torch.randn(10, 1)
    dataset = TensorDataset(Xd, Xp, y)
    return DataLoader(dataset, batch_size=2)

def test_train_one_epoch_runs():
    model = DummyModel()
    dataloader = create_dataloader()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    device = torch.device("cpu")

    loss = train.train_one_epoch(model, dataloader, criterion, optimizer, device)
    assert isinstance(loss, float)
    assert loss > 0

def test_validate_one_epoch_runs():
    model = DummyModel()
    dataloader = create_dataloader()
    criterion = nn.MSELoss()
    device = torch.device("cpu")

    loss = train.validate_one_epoch(model, dataloader, criterion, device)
    assert isinstance(loss, float)
    assert loss > 0

def test_train_model_runs():
    model = DummyModel()
    dataloader = create_dataloader()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    train.train_model(
        model=model,
        train_loader=dataloader,
        val_loader=dataloader,
        optimizer=optimizer,
        task_type="regression",
        use_logits=False,
        num_epochs=2,
        scheduler_name="step",
        scheduler_kwargs={'step_size':10},
        verbose=False
    )

def test_without_validation():
    model = DummyModel()
    dataloader = create_dataloader()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    train.train_model(
        model=model,
        train_loader=dataloader,
        val_loader=None,
        optimizer=optimizer,
        task_type="regression",
        use_logits=False,
        num_epochs=2,
        scheduler_name="step",
        scheduler_kwargs={'step_size':10},
        verbose=False
    )