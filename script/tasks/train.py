import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Callable, Literal, Optional
import numpy as np
from script.utils.scheduler import get_scheduler

def move_batch_to_device(
    Xd: torch.Tensor,
    Xp: torch.Tensor,
    y: torch.Tensor,
    device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Moves a batch of data tensors to the specified device.

    Args:
        Xd (torch.Tensor): First input tensor (e.g., drug features).
        Xp (torch.Tensor): Second input tensor (e.g., protein features).
        y (torch.Tensor): Target tensor (labels).
        device (torch.device): The device to move the tensors to.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The input tensors moved to the specified device.
    """
    return Xd.to(device), Xp.to(device), y.to(device)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): DataLoader providing training batches.
        criterion (Callable): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        device (torch.device): Device to perform computation on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    for Xd, Xp, y in dataloader:
        Xd, Xp, y = move_batch_to_device(Xd, Xp, y, device)
        optimizer.zero_grad()
        outputs = model(Xd, Xp)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate_one_epoch(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        device: torch.device) -> float:
    """
    Evaluates the model for one epoch on the validation dataset.

    Args:
        model (torch.nn.Module): The neural network model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader providing validation data batches.
        criterion (torch.nn.Module): Loss function used to compute the validation loss.
        device (torch.device): Device on which computation is performed (e.g., 'cpu' or 'cuda').

    Returns:
        float: The average loss over the entire validation dataset.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for Xd, Xp, y in dataloader:
            Xd, Xp, y = move_batch_to_device(Xd, Xp, y, device)
            outputs = model(Xd, Xp)
            loss = criterion(outputs, y)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def get_criterion(
        task_type: Literal['regression', 'binary', 'multiclass'],
        use_logits: bool = False) -> nn.modules.loss._Loss:
    """
    Returns the appropriate loss function for a given machine learning task type.

    Args:
        task_type (Literal['regression', 'binary', 'multiclass']): 
            The type of task. Must be one of 'regression', 'binary', or 'multiclass'.
        use_logits (bool, optional): 
            If True and task_type is 'binary', returns BCEWithLogitsLoss; 
            otherwise returns BCELoss for binary tasks. Ignored for other task types. 
            Default is False.

    Returns:
        nn.modules.loss._Loss: 
            The corresponding PyTorch loss function for the specified task type.

    Raises:
        ValueError: 
            If an unsupported task_type is provided.
    """
    task_type = task_type.lower()

    if task_type == "regression":
        return nn.MSELoss()
    elif task_type == "binary":
        return nn.BCEWithLogitsLoss() if use_logits else nn.BCELoss()
    elif task_type == "multiclass":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported task_type '{task_type}'. Must be 'regression', 'binary' or 'multiclass'.")

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optimizer: torch.optim.Optimizer,
    task_type: Literal['regression', 'binary', 'multiclass'],
    use_logits: bool = True,
    num_epochs: int = 50,
    scheduler_name: Optional[str] = None,
    scheduler_kwargs: Optional[dict] = None,
    verbose: bool = True
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = get_criterion(task_type, use_logits)
    scheduler = get_scheduler(optimizer, scheduler_name, **scheduler_kwargs or {})

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        if val_loader is not None:
            val_loss = validate_one_epoch(model, val_loader, criterion, device)


        if scheduler:
            if scheduler_name == 'reduceonplateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if verbose:
            if  val_loader is None:
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
