from torch import nn
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
    """
    def __init__(
            self,
            mo_params: MoleculeOrganismInteractionParams,
            ff_params: FeedForwardNetworkParams):
        super().__init__()

        self.mo = MoleculeOrganismInteraction.from_params(mo_params)
        self.ff = FeedForwardNetwork.from_params(ff_params)

    def forward(self, Xd, Xp):
        return self.ff(self.mo(Xd, Xp))
    
    
