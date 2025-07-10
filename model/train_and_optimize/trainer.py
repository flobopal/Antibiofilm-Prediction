from dataclasses import dataclass
from typing import Literal, Optional, Union
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from model.interaction import MoleculeOrganismInteractionParams
from model.decoder import FeedForwardNetworkParams
from model.full_model import FullModel

from script.tasks.train import train_model
from script.tasks.cross_validate import cross_validate_model

@dataclass
class Trainer:
    """
    Trainer class for managing the training process of a molecule-organism interaction model.
    Attributes:
        Xd (torch.Tensor): Tensor containing molecular features.
        Xp (torch.Tensor): Tensor containing organism features.
        y (torch.Tensor): Tensor containing target values.
        embed_dim (int): Embedding dimension for interaction layers.
        hidden_dims (list[int]): List of hidden layer dimensions for the feedforward network.
        activations (Union[str, list[str]]): Activation function(s) for the feedforward network. Default is 'relu'.
        num_heads (int): Number of attention heads in the interaction module. Default is 4.
        pooling (Literal['linear', 'max', 'mean']): Pooling strategy for interaction outputs. Default is 'max'.
        dropout (float): Dropout rate applied in the model. Default is 0.1.
        lr (float): Learning rate for the optimizer. Default is 0.001.
        num_epochs (int): Number of training epochs. Default is 50.
        scheduler_name (Optional[str]): Name of the learning rate scheduler to use. Default is None.
        scheduler_kwargs (Optional[dict]): Additional keyword arguments for the scheduler. Default is None.
        verbose (bool): Whether to print training progress. Default is True.
    Methods:
        get_interaction_params() -> MoleculeOrganismInteractionParams:
            Returns the parameters required to initialize the molecule-organism interaction module.
        get_ff_params() -> FeedForwardNetworkParams:
            Returns the parameters required to initialize the feedforward network.
        get_model() -> FullModel:
            Constructs and returns the full model using the specified parameters.
        get_loader() -> DataLoader:
            Creates and returns a DataLoader for the training data.
        train():
            Trains the model using the specified configuration and optimizer.
    """
    Xd: torch.Tensor
    Xp: torch.Tensor
    y: torch.Tensor
    embed_dim: int
    hidden_dims: list[int]
    activations: Union[str, list[str]] = 'relu'
    num_heads: int = 4
    pooling: Literal['linear', 'max', 'mean'] = 'max'
    dropout: float = 0.1
    lr: float = 0.001
    num_epochs: int = 50
    scheduler_name: Optional[str] = None
    scheduler_kwargs: Optional[dict] = None
    verbose: bool = True

    def get_interaction_params(self) -> MoleculeOrganismInteractionParams:
        return MoleculeOrganismInteractionParams(
            mol_input_dim=self.Xd.shape[1],
            org_input_dim=self.Xp.shape[1],
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            pooling=self.pooling,
            dropout=self.dropout
        )
    
    def get_ff_params(self) -> FeedForwardNetworkParams:
        return FeedForwardNetworkParams(
            input_dims=self.Xd.shape[1],
            hidden_dims=self.hidden_dims,
            activations=self.activations,
            dropout=self.dropout
        )
    
    def get_model(self) -> FullModel:
        return FullModel(
            self.get_interaction_params(),
            self.get_ff_params()
        )
    
    def get_loader(self) -> DataLoader:
        dataset = TensorDataset(self.Xd, self.Xp, self.y)
        return DataLoader(dataset)

    def train(self):
        train_model(
            model := self.get_model(),
            self.get_loader(),
            None,
            Adam(model.parameters(), lr = self.lr),
            'regression',
            num_epochs=self.num_epochs,
            scheduler_name=self.scheduler_name,
            scheduler_kwargs=self.scheduler_kwargs,
            verbose=self.verbose
        )

    def cross_validation(self,
                         kfolds: int = 5,
                         metrics: Optional[str]=None) -> list[float]:
        return cross_validate_model(
            FullModel,
            dict(
                mo_params = self.get_interaction_params(),
                ff_params = self.get_ff_params()
            ),
            self.Xd,
            self.Xp,
            self.y,
            Adam,
            {'lr': self.lr},
            'regression',
            k_folds=kfolds,
            num_epochs=self.num_epochs,
            scheduler_name=self.scheduler_name,
            scheduler_kwargs=self.scheduler_kwargs,
            verbose=self.verbose,
            metrics=metrics
        )