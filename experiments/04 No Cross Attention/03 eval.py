import sys
import os
from pathlib import Path
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np

sys.path.append(os.getcwd())

from script.utils.data_load import data_load
from model.decoder import FeedForwardNetwork

folder = Path("experiments", "04 No Cross Attention")

Xd, Xp, y = data_load("experiments/Antibiofilm MF+MD.csv", features_start=5, organism_encoder_path=folder / "encoder.pkl", organism_column='target_organism',
        normalizer_path=folder / "normalizer.pkl", output_column='pIC50', train_test_column='train', train_test_value='False', normalizer_start=768, normalizer_end=2000)

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

model_path = folder / "antibiofilm_model.pth"
state_dict = torch.load(model_path, weights_only=True)
model.load_state_dict(state_dict)

model.eval()
with torch.no_grad():
    y_pred = model.forward(X)

print((((y_pred - y)**2).mean())**0.5)

df = pd.read_csv("experiments/Antibiofilm MF+MD.csv", index_col = 0)
df = df[~df.train].iloc[:,:4]
df["y_pred"] = y_pred
df.to_csv(folder / "predictions.csv")