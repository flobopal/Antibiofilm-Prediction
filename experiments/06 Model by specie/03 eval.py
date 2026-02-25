import sys
import os
from pathlib import Path
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import json

sys.path.append(os.getcwd())

from script.utils.data_load import data_load
from model.decoder import FeedForwardNetwork

folder = Path("experiments", "06 Model by specie")
with open(folder / "best_params.json") as best_params_file:
    best_params = json.load(best_params_file)

predictions = pd.DataFrame()

def eval_specie(specie: str):
    csv_path = folder / "organisms" / f"{specie}.csv"
    normalizer_path = folder / "organisms" / f"normalizer {specie}.pkl"

    X, _, y = data_load(csv_path, features_start=5,
            normalizer_path=normalizer_path, output_column='pIC50',
            train_test_column='train', train_test_value='False', normalizer_start=768, normalizer_end=2000)

    X = torch.tensor(X)
    y = torch.tensor(y)

    dataset= TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64)
    model = FeedForwardNetwork(
        X.shape[1],
        **best_params[specie]["model_params"]
    )

    model_path = folder / "organisms" / "models" / f"antibiofilm_model {specie}.pth"
    state_dict = torch.load(model_path, weights_only=True)
    model.load_state_dict(state_dict)

    model.eval()
    with torch.no_grad():
        y_pred = model.forward(X)

    df_specie = pd.read_csv(csv_path, index_col=0).iloc[:,:4]
    df_specie = df_specie[~df_specie.train]
    df_specie["y_pred"] = y_pred
    global predictions
    predictions = pd.concat([predictions, df_specie], axis=0)

    print(specie, (((y_pred - y)**2).mean())**0.5)

for file in os.listdir(folder / "organisms"):
    if file.endswith('.csv'):
        eval_specie(file[:-4])

predictions.to_csv(folder / "predictions.csv")