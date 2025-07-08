from dataclasses import dataclass
from typing import Literal, Union
import torch

from model.interaction import MoleculeOrganismInteractionParams
from model.decoder import FeedForwardNetworkParams
from model.full_model import FullModel

from script.tasks.train import train_model
from script.tasks.cross_validate import cross_validate_model

@dataclass
class Trainer:
    Xd: torch.Tensor
    Xp: torch.Tensor
    embed_dim: int
    hidden_dims: list[int]
    activations: Union[str, list[str]] = 'relu'
    num_heads: int = 4
    pooling: Literal['linear', 'max', 'mean'] = 'max'
    dropout: float = 0.1

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

