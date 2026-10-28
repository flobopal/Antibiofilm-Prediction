import random

import pandas as pd
import numpy as np
import sys
import os
import shutil
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
import json

import torch
sys.path.append(os.getcwd())
from model.full_model import train_model
from script.utils.data_load import data_load
import importlib

train_file = importlib.import_module("experiments.01_model.02_train_and_eval")


DATA_PARAMS = dict(
    features_start=6,
    features_end=None,
    organism_column="target_organism",
    output_column="pIC50",
    normalizer_start=768,
    normalizer_end=900,
)

MODEL_PARAMS = {
    **DATA_PARAMS,
    **train_file.BEST_PARAMS
}

CURRENT_PATH = Path(os.path.abspath(__file__)).parent

if __name__ == "__main__":
    results = pd.DataFrame(columns=["split_number", "target_organism", "curated_smiles", "real_pIC50", "pred_pIC50"])

    for partition in range(10):
        print("=========== SPLIT",partition,"===============")
        datapath = f"dataset/split_by_scaffold/scaffold_split_{partition}.csv"
        model_path = CURRENT_PATH / f"split_{partition}"
        model_path.mkdir(exist_ok=True)
        model = train_model(
            datapath,
            organism_encoder_path=model_path / "encoder.pkl",
            normalizer_path=model_path / "normalizer.pkl",
            train_test_column="train",
            train_test_value="True",
            **MODEL_PARAMS
            )

        Xd, Xp, y_real = data_load(
            data_path=datapath,
            organism_encoder_path=model_path / "encoder.pkl",
            normalizer_path=model_path / "normalizer.pkl",
            train_test_column="train",
            train_test_value="False",
            **DATA_PARAMS
        )
        model.eval()
        with torch.no_grad():
            y_pred = model(Xd, Xp).detach().cpu().numpy()

        eval_df = pd.read_csv(datapath, index_col=0).iloc[:,:5]
        eval_df = eval_df[~eval_df.train]
        eval_df["pred_pIC50"] = y_pred
        eval_df["split_number"] = partition
        eval_df.rename(columns={"pIC50": "real_pIC50"}, inplace=True)

        results = pd.concat([results, eval_df[["split_number", "target_organism", "curated_smiles", "real_pIC50", "pred_pIC50"]]])

    results.to_csv(CURRENT_PATH / "predictions_scaffold.csv")