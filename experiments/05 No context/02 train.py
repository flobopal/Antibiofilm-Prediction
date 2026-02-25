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

folder = Path("experiments", "05 No context")

X, _, y = data_load("experiments/Antibiofilm MF+MD.csv", features_start=5, organism_encoder_path=folder / "encoder.pkl", organism_column='target_organism',
        normalizer_path=folder / "normalizer.pkl", output_column='pIC50', train_test_column='train', train_test_value='True', normalizer_start=768, normalizer_end=2000)

X = torch.tensor(X)
y = torch.tensor(y)

dataset= TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=64)
model = FeedForwardNetwork(
    X.shape[1],
    [2**6, 2**6],
    ["identity", "sigmoid", "elu"],
    0.0832
)
optimizer = Adam(model.parameters(), lr=0.00177)
train_model(
    model,
    loader,
    None,
    optimizer,
    "regression",
    num_epochs=2**7,
    scheduler_name="step",
    scheduler_kwargs={"step_size": 20, "gamma": 0.635}
)

torch.save(model.state_dict(), folder / "antibiofilm_model.pth")