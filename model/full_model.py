from torch import nn
import torch
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

    
