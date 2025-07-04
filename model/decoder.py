from typing import List, Self, Union
import torch.nn as nn
from script.utils.activation_functions import get_activation
from dataclasses import dataclass

@dataclass
class FeedForwardNetworkParams:
    input_dims: int
    hidden_dims: list[int]
    activations: Union[str, List[str]]
    dropout: float = 0.1

class FeedForwardNetwork(nn.Module):
    """
    A configurable feed-forward neural network module.

    Args:
        input_dim (int): The size of the input features.
        layer_dims (list[int]): A list specifying the number of units in each hidden layer.
        activations (str or list[str]): The activation function(s) to use for each hidden layer.
            If a single string is provided, the same activation is used for all layers.
            If a list, its length must match `layer_dims`.
        dropout (float, optional): Dropout probability applied after each activation. Default is 0.1.

    Attributes:
        model (nn.Sequential): The sequential container of layers forming the feed-forward network.

    Raises:
        AssertionError: If the length of `activations` does not match the length of `layer_dims`.

    Forward Input:
        x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

    Forward Output:
        torch.Tensor: Output tensor of shape (batch_size, 1).

    Example:
        >>> net = FeedForwardNetwork(input_dim=128, layer_dims=[64, 32], activations=['relu', 'tanh'])
        >>> output = net(torch.randn(16, 128))
    """
    def __init__(self,
                 input_dim: int,
                 layer_dims: list[int],
                 activations: str|list[str],
                 dropout: float=0.1):
        super().__init__()
        if isinstance(activations, str):
            activations = [activations]*len(layer_dims)
        assert len(layer_dims) == len(activations), "length of activations should match length of layers"

        layers = []
        dim = input_dim

        for next_dim, act_name in zip(layer_dims, activations):
            layers.append(get_activation(act_name))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(dim, next_dim))
            dim = next_dim

        layers.append(nn.Linear(dim, 1))

        self.model = nn.Sequential(*layers)

    @classmethod
    def from_params(cls, params: FeedForwardNetworkParams) -> Self:
        """
        Create a FeedForwardNetwork instance from FeedForwardNetworkParams.

        Args:
            params (FeedForwardNetworkParams): Parameters for the feed-forward network.

        Returns:
            FeedForwardNetwork: An instance of the feed-forward network.
        """
        return cls(
                input_dim=params.input_dims,
                layer_dims=params.hidden_dims,
                activations=params.activations,
                dropout=params.dropout
            )

    def forward(self, x):
        return self.model(x)