import gc
import os
import pandas as pd
import torch
import numpy as np
from typing import Callable, Optional

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from script.tasks.train import train_model, get_criterion, validate_one_epoch
from script.utils.metrics import evaluate
from script.utils.data_load import get_features_array
from script.utils.weights import organism_loss_weights


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def clean():
    """
    Performs memory cleanup for both CPU and GPU.

    This function triggers Python's garbage collector to free up unreferenced memory.
    If a CUDA-capable GPU is available, it also clears the CUDA memory cache and
    collects inter-process communication (IPC) resources to help prevent memory leaks
    and fragmentation during intensive GPU computations.
    """
    gc.collect()
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def cross_validate(
    dataset: pd.DataFrame,
    model_factory: Callable[[int, int], torch.nn.Module],
    features_start: int | str,
    output_column: str,
    optimizer_class: type[torch.optim.Optimizer],
    optimizer_kwargs: dict,
    task_type: str,
    fold_column: str = "kfold",
    features_end: int | str | None = None,
    normalize_features: bool = True,
    normalizer_start: int | None = None,
    normalizer_end: int | None = None,
    use_logits: bool = True,
    num_epochs: int = 50,
    scheduler_name: str = None,
    scheduler_kwargs: Optional[dict] = None,
    batch_size: int = 64,
    verbose: bool = True,
    metrics: Optional[str] = None,
    after_fold_callback: Optional[Callable[[int, float], None]] = None,
    include_organism_features: bool = True,
    balance_organism: bool = False,
    organism_column: Optional[str] = None,
) -> list[float]:
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

    fold_labels = dataset[fold_column]
    fold_values = fold_labels.unique()
    k_folds = len(fold_values)


    required_columns = [organism_column, output_column]
    missing = [column for column in required_columns if column not in dataset]
    if missing:
        raise KeyError(f"Columns not found in dataset: {missing}")

    raw_features = np.asarray(
        get_features_array(dataset, features_start, features_end),
        dtype=np.float32,
    )
    targets = dataset[output_column].to_numpy(dtype=np.float32)
    if organism_column is not None:
        organisms = dataset[organism_column].to_numpy()

    values = []
    fold_iterator = (
        (np.flatnonzero(fold_labels != fold_value), np.flatnonzero(fold_labels == fold_value))
        for fold_value in fold_values
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold, (train_idx, val_idx) in enumerate(fold_iterator):
        clean()
        if verbose:
            print(f"\n--- Fold {fold + 1}/{k_folds} ---")

        Xd_train = raw_features[train_idx].copy()
        Xd_val = raw_features[val_idx].copy()
        if normalize_features:
            scaler = StandardScaler()
            train_slice = Xd_train[:, normalizer_start:normalizer_end]
            val_slice = Xd_val[:, normalizer_start:normalizer_end]
            Xd_train[:, normalizer_start:normalizer_end] = (
                scaler.fit_transform(train_slice)
            )
            Xd_val[:, normalizer_start:normalizer_end] = scaler.transform(
                val_slice
            )

        if include_organism_features:
            if organism_column is None:
                raise ValueError("organism_column must be provided if include_organism_features is True")
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                dtype=np.float32,
            )
            Xp_train = encoder.fit_transform(organisms[train_idx, None])
            Xp_val = encoder.transform(organisms[val_idx, None])
            train_inputs = (
                torch.from_numpy(Xd_train),
                torch.from_numpy(Xp_train),
            )
            val_inputs = (
                torch.from_numpy(Xd_val),
                torch.from_numpy(Xp_val),
            )
            model = model_factory(Xd_train.shape[1], Xp_train.shape[1])
        else:
            train_inputs = (torch.from_numpy(Xd_train),)
            val_inputs = (torch.from_numpy(Xd_val),)
            model = model_factory(Xd_train.shape[1], 0)

        train_tensors = (*train_inputs, torch.from_numpy(targets[train_idx]))
        val_tensors = (*val_inputs, torch.from_numpy(targets[val_idx]))
        if balance_organism:
            if organism_column is None:
                raise ValueError("organism_column must be provided if balance_organism is True")
            train_tensors += (organism_loss_weights(organisms[train_idx]),)
            val_tensors += (organism_loss_weights(organisms[val_idx]),)
        train_dataset = TensorDataset(*train_tensors)
        val_dataset = TensorDataset(*val_tensors)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            task_type=task_type,
            use_logits=use_logits,
            num_epochs=num_epochs,
            scheduler_name=scheduler_name,
            scheduler_kwargs=scheduler_kwargs,
            verbose=verbose,
            weighted_loss=balance_organism,
            weighted_val_loss=balance_organism,
        )
        model.eval()
        if metrics is not None:
            validation_tensors = tuple(map(torch.cat, zip(*val_loader)))
            weights = None
            if balance_organism:
                *X_test, y_test, weights = validation_tensors
            else:
                *X_test, y_test = validation_tensors
            y_pred = model(*(X.to(device) for X in X_test))
            validation_value = evaluate(
                metrics, y_test, y_pred, weights
            )
        else:
            criterion = get_criterion(
                task_type, use_logits, reduction="none" if balance_organism else "mean"
            )
            validation_value = validate_one_epoch(
                model, val_loader, criterion, device, weighted_loss=balance_organism
            )
        values.append(validation_value)

        if verbose:
            metric_name = metrics if metrics is not None else "loss"
            print(
                f"Fold {fold + 1} final validation {metric_name}: "
                f"{validation_value:.4f}"
            )

        if after_fold_callback is not None:
            after_fold_callback(fold, validation_value)

    return values