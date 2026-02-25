import sys
import os
from pathlib import Path
import torch
import pandas as pd

sys.path.append(os.getcwd())

from script.utils.data_load import data_load
from model.full_model import FullModel

folder = Path("experiments", "01 Model Train and evaluation")

Xd, Xp, y = data_load("experiments/Antibiofilm MF+MD.csv", features_start=5, organism_encoder_path=folder / "encoder.pkl", organism_column='target_organism',
          normalizer_path=folder / "normalizer.pkl", output_column='pIC50', train_test_column='train', train_test_value='False', normalizer_start=768, normalizer_end=2000)

model_path = folder / "antibiofilm_model.pth"
model = FullModel.load(model_path)

model.eval()
with torch.no_grad():
    y_pred = model.forward(Xd, Xp)

print((((y_pred - y)**2).mean())**0.5)

df = pd.read_csv("experiments/Antibiofilm MF+MD.csv", index_col = 0)
df = df[~df.train].iloc[:,:4]
df["y_pred"] = y_pred
df.to_csv(folder / "predictions.csv")