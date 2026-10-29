import numpy as np
import pandas as pd
import torch

def organism_loss_weights(organisms) -> torch.Tensor:
    """Return N / (K * count[organism]), using only the supplied rows."""
    if isinstance(organisms, torch.Tensor):
        organisms = organisms.detach().cpu().numpy()
    labels = np.asarray(organisms)
    if labels.ndim != 1 or len(labels) == 0 or pd.isna(labels).any():
        raise ValueError("organisms must be nonempty, one-dimensional, nonmissing labels")
    codes, unique = pd.factorize(labels)
    counts = np.bincount(codes)
    return torch.tensor(len(labels) / (len(unique) * counts[codes]), dtype=torch.float32)