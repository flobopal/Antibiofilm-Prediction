import sys
import os
sys.path.append(os.getcwd())

from pathlib import Path
import numpy as np
import json

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam

from script.utils.data_load import data_load

from model.decoder import FeedForwardNetwork

from script.tasks.train import train_model

folder = Path("experiments", "06 Model by specie")
with open(folder / "best_params.json") as best_params_file:
    best_params = json.load(best_params_file)

def train_specie(specie: str):
    print(f"=========================={specie.upper()}==============================")
    csv_path = folder / "organisms" / f"{specie}.csv"
    normalizer_path = folder / "organisms" / f"normalizer {specie}.pkl"
    X, _, y = data_load(csv_path, features_start=5,
            normalizer_path=normalizer_path, output_column='pIC50',
            train_test_column='train', train_test_value='True', normalizer_start=768, normalizer_end=2000)

    X = torch.tensor(X)
    y = torch.tensor(y)

    specie_params = best_params[specie]

    dataset= TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64)
    model = FeedForwardNetwork(
        X.shape[1],
        **specie_params["model_params"]
    )
    optimizer = Adam(model.parameters(), lr=specie_params["lr"])
    train_model(
        model,
        loader,
        None,
        optimizer,
        "regression",
        **specie_params["trainer_params"]
    )

    torch.save(model.state_dict(), folder / "organisms" / "models" / f"antibiofilm_model {specie}.pth")

for file in os.listdir(folder / "organisms"):
    if file.endswith('.csv'):
        train_specie(file[:-4])

