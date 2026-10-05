from typing import Optional

from torch import nn
import torch
from script.utils.data_load import data_load
from script.tasks.train import train_model as _train_model
from model.interaction import MoleculeOrganismInteraction, MoleculeOrganismInteractionParams
from model.decoder import FeedForwardNetwork, FeedForwardNetworkParams

class FullModel(nn.Module):
    """
    FullModel combines a molecule-organism interaction module with a feedforward network for prediction tasks.

    Args:
        mo_params (MoleculeOrganismInteractionParams): Parameters for initializing the molecule-organism interaction module.
        ff_params (FeedForwardNetworkParams): Parameters for initializing the feedforward network.

    Attributes:
        mo (MoleculeOrganismInteraction): Module modeling interactions between molecules and organisms.
        ff (FeedForwardNetwork): Feedforward neural network for downstream prediction.

    Methods:
        forward(Xd, Xp):
            Processes input features through the molecule-organism interaction module followed by the feedforward network.

            Args:
                Xd: Input features representing molecules.
                Xp: Input features representing organisms.

            Returns:
                Output of the feedforward network after processing the interaction features.

        save(path):
            Saves the model state and its hyperparameters to a file.

        load(path):
            Loads the model from a saved file including its parameters and weights.
    """
    def __init__(
            self,
            mo_params: MoleculeOrganismInteractionParams,
            ff_params: FeedForwardNetworkParams):
        super().__init__()

        self.mo = MoleculeOrganismInteraction.from_params(mo_params)
        self.ff = FeedForwardNetwork.from_params(ff_params)
        self.mo_params = mo_params
        self.ff_params = ff_params

    def forward(self, Xd, Xp):
        return self.ff(self.mo(Xd, Xp))
    
    def save(self, path: str):
        torch.save({
            "model_state_dict": self.state_dict(),
            "mo_params": self.mo_params.asdict(),
            "ff_params": self.ff_params.asdict()
        }, path)
    
    @classmethod
    def load(cls, path: str):
        checkpoint = torch.load(path, weights_only=True)
        mo_params = MoleculeOrganismInteractionParams.fromdict(checkpoint["mo_params"])
        ff_params = FeedForwardNetworkParams.fromdict(checkpoint["ff_params"])

        model = cls(mo_params, ff_params)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    
def train_model(
        data_path: str,
        features_start: int | str = 0,
        features_end: Optional[int | str] = None,
        organism_encoder_path: Optional[str] = None,
        organism_column: Optional[str | int] = None,
        normalizer_path: Optional[str] = None,
        normalizer_start: Optional[int] = None,
        normalizer_end: Optional[int] = None,
        output_column: Optional[int | str] = None,
        train_test_column: Optional[str] = None,
        train_test_value: Optional[str] = None,
        embed_dim: int = 2**4,
        hidden_dims: list[int] =[2**5, 2**9],
        activations: list[str] = ["tanh", "tanh", "leaky_relu"],
        dropout: float = 0.0311,
        lr: float = 9.27e-5,
        num_epochs: int = 2**8,
        batch_size: int = 64,
        scheduler_name: Optional[str] = None,
        scheduler_kwargs: Optional[dict] = None,
        last_activation_params: Optional[dict] = None,
        save_path: Optional[str] = None,
        verbose: bool = True
) -> FullModel:
    Xd, Xp, y = data_load(
        data_path,
        features_start=features_start,
        features_end=features_end,
        organism_encoder_path=organism_encoder_path,
        organism_column=organism_column,
        normalizer_path=normalizer_path,
        normalizer_start=normalizer_start,
        normalizer_end=normalizer_end,
        output_column=output_column,
        train_test_column=train_test_column,
        train_test_value=train_test_value
    )
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xd, Xp, y),
        batch_size=batch_size,
        shuffle=True
    )
    model = FullModel(
        mo_params=MoleculeOrganismInteractionParams(
            mol_input_dim=Xd.shape[1],
            org_input_dim=Xp.shape[1],
            embed_dim=embed_dim,
            dropout=dropout
        ),
        ff_params=FeedForwardNetworkParams(
            input_dims=embed_dim,
            hidden_dims=hidden_dims,
            activations=activations,
            dropout=dropout
        )
    )
    _train_model(
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