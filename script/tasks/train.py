import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Callable, Literal, Optional
from tqdm import tqdm
from script.utils.scheduler import get_scheduler

def move_batch_to_device(
    X: list[torch.Tensor],
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
    output = [x.to(device) for x in X]
    output.append(y.to(device))
    return output


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    weighted_loss: bool = False
) -> float:
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): DataLoader providing training batches.
        criterion (Callable): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        device (torch.device): Device to perform computation on.
        weighted_loss (bool, optional): Whether to use weighted loss. Default is False.
    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in dataloader:
        if weighted_loss:
            *X, y, weights = batch
            weights = weights.to(device)
        else:
            *X, y = batch
        *X, y = move_batch_to_device(X, y, device)
        optimizer.zero_grad()
        outputs = model(*X)
        loss = criterion(outputs, y)
        if weighted_loss:
            loss = (loss.reshape(len(y), -1).mean(dim=1) * weights).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        total_samples += len(y)
    return total_loss / total_samples


def validate_one_epoch(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        device: torch.device,
        weighted_loss: bool = False
        ) -> float:
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
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            if weighted_loss:
                *X, y, weights = batch
                weights = weights.to(device)
            else:
                *X, y = batch
            *X, y = move_batch_to_device(X, y, device)
            outputs = model(*X)
            loss = criterion(outputs, y)
            if weighted_loss:
                loss = (loss.reshape(len(y), -1).mean(dim=1) * weights).mean()
            total_loss += loss.item() * len(y)
            total_samples += len(y)
    return total_loss / total_samples

def get_criterion(
        task_type: Literal['regression', 'binary', 'multiclass'],
        use_logits: bool = False,
        reduction: str = 'mean'
) -> nn.modules.loss._Loss:
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
        return nn.MSELoss(reduction=reduction)
    elif task_type == "binary":
        return nn.BCEWithLogitsLoss(reduction=reduction) if use_logits else nn.BCELoss(reduction=reduction)
    elif task_type == "multiclass":
        return nn.CrossEntropyLoss(reduction=reduction)
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
    weighted_loss: bool = False,
    weighted_val_loss: bool = False,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    train_criterion = get_criterion(task_type, use_logits, reduction="none" if weighted_loss else "mean")
    val_criterion = get_criterion(task_type, use_logits, reduction="none" if weighted_val_loss else "mean")
    scheduler = get_scheduler(optimizer, scheduler_name, **scheduler_kwargs or {})
    tqdm_bar = tqdm(range(num_epochs), desc="Training Progress", unit="epoch", colour="blue")
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, train_criterion, optimizer, device)
        if val_loader is not None:
            val_loss = validate_one_epoch(model, val_loader, val_criterion, device)

        if scheduler:
            if scheduler_name == 'reduceonplateau':
                scheduler.step(val_loss if val_loader is not None else train_loss)
            else:
                scheduler.step()
        if val_loader is not None:
            tqdm_bar.set_postfix({
                "train_loss": train_loss,
                "val_loss": val_loss
            })
        else:
            tqdm_bar.set_postfix({
            "train_loss": train_loss,
        })
