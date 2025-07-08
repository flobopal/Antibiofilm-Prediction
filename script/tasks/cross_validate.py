from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader, Subset
import torch
import numpy as np
from script.tasks.train import train_model, get_criterion, validate_one_epoch

def cross_validate_model(
    
    model_class: type,
    model_kwargs: dict,
    Xd: torch.Tensor,
    Xp: torch.Tensor,
    y: torch.Tensor,
    optimizer_class: torch.optim.Optimizer,
    optimizer_kwargs: dict,
    task_type: str,
    use_logits: bool = True,
    k_folds: int = 5,
    num_epochs: int = 50,
    scheduler_name: str = None,
    batch_size: int = 64,
    verbose: bool = True
):
    """
    Performs K-fold cross-validation on a given PyTorch model.

    Args:
        model_class (type): The class of the model to instantiate for each fold.
        model_kwargs (dict): Dictionary of keyword arguments to initialize the model.
        Xd (torch.Tensor): Input tensor for the first modality (e.g., drugs).
        Xp (torch.Tensor): Input tensor for the second modality (e.g., organisms).
        y (torch.Tensor): Target tensor.
        optimizer_class (torch.optim.Optimizer): Optimizer class to use for training.
        optimizer_kwargs (dict): Dictionary of keyword arguments for the optimizer.
        task_type (str): Type of task, e.g., 'regression' or 'classification'.
        use_logits (bool, optional): Whether the model outputs logits. Defaults to True.
        k_folds (int, optional): Number of folds for cross-validation. Defaults to 5.
        num_epochs (int, optional): Number of training epochs per fold. Defaults to 50.
        scheduler_name (str, optional): Name of the learning rate scheduler to use. Defaults to None.
        batch_size (int, optional): Batch size for data loaders. Defaults to 64.
        verbose (bool, optional): Whether to print progress information. Defaults to True.

    Returns:
        list: A list containing the validation loss for each fold.
    """
    dataset = TensorDataset(Xd, Xp, y)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    val_losses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(Xd)):
        if verbose:
            print(f"\n--- Fold {fold + 1}/{k_folds} ---")

        # Dataloaders
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Initialize new model
        model = model_class(**model_kwargs)
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

        # Training
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            task_type=task_type,
            use_logits=use_logits,
            num_epochs=num_epochs,
            scheduler_name=scheduler_name,
            verbose=verbose
        )

        # Final fold validation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        criterion = get_criterion(task_type, use_logits)
        val_loss = validate_one_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)

    return val_losses