from typing import Literal, Self
import torch
import torch.nn as nn

from dataclasses import dataclass, asdict

@dataclass
class MoleculeOrganismInteractionParams:
    mol_input_dim: int
    org_input_dim: int
    embed_dim: int
    dropout: float = 0.1

    def asdict(self):
        return asdict(self)
    
    @classmethod
    def fromdict(cls, dict):
        return cls(**dict)


class MoleculeOrganismInteraction(nn.Module):
    """Compact molecular features with residual, organism-conditioned FiLM.

    Inputs have shapes (batch, fd) and (batch, fo); output is (batch, fk).
    Zero initialization starts with molecular features alone. The learned
    multiplicative factor 1 + tanh(gamma) is bounded between zero and two;
    beta supplies an additive organism effect. Heads and pooling are unused.
    """

    def __init__(self, fd: int, fo: int, fk: int, dropout: float = 0.1):
        super().__init__()
        self.molecule_encoder = nn.Sequential(
            nn.Linear(fd, fk),
            nn.LayerNorm(fk),
            nn.SiLU(),
        )
        self.organism_encoder = nn.Linear(fo, 2 * fk)
        nn.init.zeros_(self.organism_encoder.weight)
        nn.init.zeros_(self.organism_encoder.bias)
        self.dropout = nn.Dropout(dropout)

    @classmethod
    def from_params(cls, params: MoleculeOrganismInteractionParams) -> Self:
        return cls(params.mol_input_dim, params.org_input_dim,
                   params.embed_dim, params.dropout)

    def forward(self, Xd: torch.Tensor, Xo: torch.Tensor) -> torch.Tensor:
        h = self.molecule_encoder(Xd)
        gamma, beta = self.organism_encoder(Xo).chunk(2, dim=-1)
        return self.dropout((1 + torch.tanh(gamma)) * h + beta)
