from typing import List, Optional, Self, Union
import torch
import torch.nn as nn
from script.utils.activation_functions import get_activation
from dataclasses import dataclass, asdict

from script.utils.data_load import data_load
from script.tasks.train import train_model

@dataclass
class FeedForwardNetworkParams:
    input_dims: int
    hidden_dims: list[int]
    activations: Union[str, List[str]]
    dropout: float = 0.1
    last_activation_params: Optional[dict] = None

    def asdict(self):
        return asdict(self)
    
    @classmethod
    def fromdict(cls, dict):
        return cls(**dict)

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
                 dropout: float=0.1,
                 last_activation_params: Optional[dict] = None):
        super().__init__()
        if isinstance(activations, str):
            activations = [activations]*(len(layer_dims)+1)
        assert len(layer_dims)+1 == len(activations), "length of activations should match length of layers"

        layers = []
        dim = input_dim

        #Add hidden layers
        for next_dim, act_name in zip(layer_dims, activations):
            layers.append(get_activation(act_name))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(dim, next_dim))
            dim = next_dim

        #Add output layer
        layers.append(get_activation(activations[-1], last_activation_params))
        layers.append(nn.Dropout(dropout))
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
                dropout=params.dropout,
                last_activation_params=params.last_activation_params
        )

    def forward(self, x, x2 = None):
        if x2 is not None:
            x = torch.cat((x, x2), 1)
        return self.model(x).squeeze(-1)

def train_mlp(
        data_path: str,
        features_start: int | str = 0,
        features_end: Optional[int | str] = None,
        organism_encoder_path: Optional[str] = None,
        organism_column: Optional[str | int] = None,
        organism_filter: Optional[str] = None,
        normalizer_path: Optional[str] = None,
        normalizer_start: Optional[int] = None,
        normalizer_end: Optional[int] = None,
        output_column: Optional[int | str] = None,
        train_test_column: Optional[str] = None,
        train_test_value: Optional[str] = None,
        hidden_dims: list[int] =[2**6],
        activations: list[str] = ["relu", "relu"],
        dropout: float = 0.5,
        lr: float = 1e-5,
        num_epochs: int = 2**8,
        batch_size: int = 64,
        scheduler_name: Optional[str] = None,
        scheduler_kwargs: Optional[dict] = None,
        last_activation_params: Optional[dict] = None,
        save_path: Optional[str] = None
) -> FeedForwardNetwork:
    Xd, Xp, y = data_load(
        data_path,
        features_start=features_start,
        features_end=features_end,
        organism_encoder_path=organism_encoder_path,
        organism_column=organism_column,
        organism_filter=organism_filter,
        normalizer_path=normalizer_path,
        normalizer_start=normalizer_start,
        normalizer_end=normalizer_end,
        output_column=output_column,
        train_test_column=train_test_column,
        train_test_value=train_test_value
    )
    print(Xd, ".........", Xp, ".........", y, sep='\n')

    input_dim = Xd.shape[1]
    if Xp is None:
        tensors = (Xd, y)
    else:
        tensors = (Xd, Xp, y)
        input_dim += Xp.shape[1]
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=True
    )
    model = FeedForwardNetwork(
        input_dim=input_dim,
        layer_dims=hidden_dims,
        activations=activations,
        dropout=dropout,
        last_activation_params=last_activation_params
    )
    train_model(
        model=model,
        train_loader=train_loader,
        task_type='regression',
        optimizer=torch.optim.Adam(model.parameters(), lr=lr),
        num_epochs=num_epochs,
        scheduler_name=scheduler_name,
        scheduler_kwargs=scheduler_kwargs,
    )

    if save_path is not None:
        model.save(save_path)

    return model