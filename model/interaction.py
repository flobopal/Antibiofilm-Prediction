import torch
import torch.nn as nn

class MoleculeOrganismInteraction(nn.Module):
    """
    MoleculeOrganismInteraction implements a custom multi-head attention mechanism for modeling interactions between molecule and organism representations.
    Args:
        fd (int): Feature dimension of the molecule input (Xd).
        fo (int): Feature dimension of the organism input (Xp).
        fk (int): Feature dimension for the attention keys/queries.
        num_heads (int, optional): Number of attention heads. Default is 4.
        dropout (float, optional): Dropout probability applied after the output projection. Default is 0.1.
    Attributes:
        num_heads (int): Number of attention heads.
        fk (int): Feature dimension for keys/queries.
        fd (int): Feature dimension of molecule input.
        Wq (nn.ModuleList): List of linear layers projecting organism features to queries for each head.
        Wk (nn.ModuleList): List of linear layers projecting molecule features to keys for each head.
        Wv (nn.ModuleList): List of linear layers projecting molecule features to values for each head.
        output_proj (nn.Linear): Linear layer projecting concatenated head outputs to final output dimension.
        dropout (nn.Dropout): Dropout layer applied after output projection.
    Forward Args:
        Xd (torch.Tensor): Molecule input tensor of shape (n, fd).
        Xp (torch.Tensor): Organism input tensor of shape (n, fo).
    Returns:
        torch.Tensor: Output tensor of shape (n, fd), representing the fused interaction features.
    Notes:
        - This module computes attention weights as the dot product between projected queries and keys, without applying softmax normalization.
        - The attention weights are used to scale the projected values, and outputs from all heads are concatenated and projected to the final output dimension.
        - Designed for use in tasks such as regression or classification after further processing.
    """
    def __init__(self, fd, fo, fk, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.fk = fk
        self.fd = fd
        
        # Projections for each head
        self.Wq = nn.ModuleList([nn.Linear(fo, fk) for _ in range(num_heads)])
        self.Wk = nn.ModuleList([nn.Linear(fd, fk) for _ in range(num_heads)])
        self.Wv = nn.ModuleList([nn.Linear(fd, fd) for _ in range(num_heads)])

        # Post-fusion
        self.output_proj = nn.Linear(num_heads * fd, fd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Xd, Xp):
        # Accept (n, fd) and (n, fo), convert to (n, 1, fd)/(n, 1, fo)
        if Xd.dim() == 2:
            Xd = Xd.unsqueeze(1)  # (n, 1, fd)
        if Xp.dim() == 2:
            Xp = Xp.unsqueeze(1)  # (n, 1, fo)
        heads_output = []

        for i in range(self.num_heads):
            Q = self.Wq[i](Xp)  # (n, 1, fk)
            K = self.Wk[i](Xd)  # (n, 1, fk)
            V = self.Wv[i](Xd)  # (n, 1, fd)

            # Dot product without softmax
            attn_weight = torch.sum(Q * K, dim=-1, keepdim=True)  # (n, 1, 1)
            Z = attn_weight * V  # (n, 1, fd)

            heads_output.append(Z)

        # Concatenate outputs per channel
        Z_concat = torch.cat(heads_output, dim=-1)  # (n, 1, num_heads * fd)

        # Final projection
        Z_out = self.output_proj(Z_concat)  # (n, 1, fd)
        Z_out = self.dropout(Z_out)

        return Z_out.squeeze(1)  # (n, fd)

