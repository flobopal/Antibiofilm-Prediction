import sys
import os
sys.path.append(os.getcwd())

from pathlib import Path
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam

from script.utils.data_load import data_load

from model.decoder import FeedForwardNetwork

from script.tasks.train import train_model

folder = Path("experiments", "04 No Cross Attention")

Xd, Xp, y = data_load("experiments/Antibiofilm MF+MD.csv", features_start=5, organism_encoder_path=folder / "encoder.pkl", organism_column='target_organism',
        normalizer_path=folder / "normalizer.pkl", output_column='pIC50', train_test_column='train', train_test_value='True', normalizer_start=768, normalizer_end=2000)

X = np.concatenate((Xd, Xp), axis=1)
X = torch.tensor(X)
y = torch.tensor(y)

dataset= TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=64)
model = FeedForwardNetwork(
    X.shape[1],
    [2**6, 2**10],
    ["gelu", "gelu", "identity"],
    0.1497
)
optimizer = Adam(model.parameters(), lr=0.001686)
train_model(
    model,
    loader,
    None,
    optimizer,
    "regression",
    num_epochs=2**7
)

torch.save(model.state_dict(), folder / "antibiofilm_model.pth")