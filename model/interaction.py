from typing import Literal, Self
import torch
import torch.nn as nn

from dataclasses import dataclass, asdict

@dataclass
class MoleculeOrganismInteractionParams:
    mol_input_dim: int
    org_input_dim: int
    embed_dim: int
    num_heads: int = 4
    pooling: Literal['linear', 'max', 'mean'] = 'max'
    dropout: float = 0.1

    def asdict(self):
        return asdict(self)
    
    @classmethod
    def fromdict(cls, dict):
        return cls(**dict)


class MoleculeOrganismInteraction(nn.Module):
    """
    'MoleculeOrganismInteraction' implements a custom multi-head attention mechanism for modeling interactions between molecule and organism representations.

    Args:
        fd (int): Feature dimension of the molecule input (Xd).
        fo (int): Feature dimension of the organism input (Xp).
        fk (int): Feature dimension for the attention keys/queries.
        num_heads (int, optional): Number of attention heads. Default is 4.
        dropout (float, optional): Dropout probability applied after the output projection. Default is 0.1.
        pooling (Literal['linear', 'max', 'mean'], optional): Pooling strategy applied to the concatenated head outputs.
            - 'linear': Applies a linear projection to the concatenated outputs.
            - 'max': Applies max pooling across the head outputs.
            - 'mean': Applies mean pooling across the head outputs.
            Default is 'max'.

    Attributes:
        num_heads (int): Number of attention heads.
        fk (int): Feature dimension for keys/queries.
        fd (int): Feature dimension of molecule input.
        pooling (str): Pooling strategy used for combining head outputs.
        Wq (nn.ModuleList): List of linear layers projecting organism features to queries for each head.
        Wk (nn.ModuleList): List of linear layers projecting molecule features to keys for each head.
        Wv (nn.ModuleList): List of linear layers projecting molecule features to values for each head.
        output_proj (nn.Linear, optional): Linear layer projecting concatenated head outputs to final output dimension (used if pooling='linear').
        dropout (nn.Dropout): Dropout layer applied after output projection or pooling.

    Forward Args:
        Xd (torch.Tensor): Molecule input tensor of shape (n, fd).
        Xp (torch.Tensor): Organism input tensor of shape (n, fo).

    Returns:
        torch.Tensor: Output tensor of shape (n, fd), representing the fused interaction features.

    Notes:
        - This module computes attention weights as the dot product between projected queries and keys, without applying softmax normalization.
        - The attention weights are used to scale the projected values, and outputs from all heads are combined using the specified pooling strategy.
        - Designed for use in tasks such as regression or classification after further processing.
    """
    def __init__(
            self,
            fd, fo, fk,
            num_heads=4,
            dropout=0.1,
            pooling: Literal['linear', 'max', 'mean'] = 'max'):
        
        assert pooling in ["linear", "mean", "max"], f"Unknown pooling: {pooling}"

        super().__init__()
        self.fd = fd
        self.fo = fo
        self.fk = fk
        self.num_heads = num_heads
        self.pooling = pooling.lower()
        
        # Projections
        self.Wq = nn.Linear(fo, fk * num_heads, bias=True)
        self.Wk = nn.Linear(fd, fk * num_heads, bias=True)
        self.Wv = nn.Linear(fd, fd * num_heads, bias=True)

        # Post-fusion
        if self.pooling == 'linear':
            self.output_proj = nn.Linear(num_heads * fd, fd)
        self.dropout = nn.Dropout(dropout)

    @classmethod
    def from_params(cls, params: MoleculeOrganismInteractionParams) -> Self:
        """
        Creates an instance of the class from a MoleculeOrganismInteractionParams object.

        Args:
            params (MoleculeOrganismInteractionParams): An object containing the configuration parameters
                required to initialize the class, including molecular input dimension, organism input
                dimension, embedding dimension, number of attention heads, dropout rate, and pooling type.

        Returns:
            Self: An instance of the class initialized with the parameters provided in `params`.
        """
        return cls(
            fd=params.mol_input_dim,
            fo=params.org_input_dim,
            fk=params.embed_dim,
            num_heads=params.num_heads,
            dropout=params.dropout,
            pooling=params.pooling
        )

    def forward(self, Xd, Xo):
        # Xd: (batch, 1, fd)  molecule embeddings
        # Xo: (batch, 1, fo)  organism embeddings

        batch_size = Xd.size(0)

        Q = self.Wq(Xo).view(batch_size, self.num_heads, self.fk)   # (batch, heads, fk)
        K = self.Wk(Xd).view(batch_size, self.num_heads, self.fk)   # (batch, heads, fk)
        V = self.Wv(Xd).view(batch_size, self.num_heads, self.fd)   # (batch, heads, fd)

        attn_scores = (Q * K).sum(dim=2, keepdim=True)  # (batch, heads, 1)
        Z = attn_scores * V                             # (batch, heads, fd)

        if self.pooling == "linear":
            Z = Z.view(batch_size, -1)                  # (batch, heads * fd)
            Z = self.output_proj(Z)                     # (batch, fd)
        elif self.pooling == "mean":
            Z = Z.mean(dim=1)                           # (batch, fd)
        elif self.pooling == "max":
            Z, _ = Z.max(dim=1)                         # (batch, fd)

        return self.dropout(Z)

