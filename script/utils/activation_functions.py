from typing import Callable, Optional
import torch.nn as nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "identity": nn.Identity,
}

def get_activation(name : str, params: Optional[dict] = None) -> Callable:
    """
    Retrieves an activation function by name.
    Args:
        name (str): The name of the activation function to retrieve.
    Returns:
        Callable: The activation function corresponding to the given name.
    Raises:
        ValueError: If the specified activation function name is not recognized.
    """
    params = params or {}
    if name not in ACTIVATIONS:
        raise ValueError(f"Activation '{name}' not recognized")
    return ACTIVATIONS[name](**params)

def list_names() -> list[str]:
    """
    Returns a list of available activation function names.

    Returns:
        list[str]: List of activation function names.
    """
    return list(ACTIVATIONS.keys())
    