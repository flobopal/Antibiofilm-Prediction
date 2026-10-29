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
JSON_PATH = CURRENT_PATH / "results.json"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

def evaluate_specie(data, species):
    general_results = {}
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, "r") as file:
            general_results = json.load(file)
    results = general_results.setdefault(species, {})
    specific_train_df = data[data.train & data.target_organism.eq(species)]
    general_train_df = data[
        data.train
        & data.target_organism.ne(species)
        & ~data.curated_smiles.isin(specific_train_df.curated_smiles)]
    test_df = data[~data.train & data.target_organism.eq(species)]

    temp_path = CURRENT_PATH / "temp"
    shutil.rmtree(temp_path, ignore_errors=True)

    for k in map(round,np.linspace(4, min(64,len(specific_train_df)), 10)):
        print("------------", k, "--------------")
        k_results = results.setdefault(str(k), {})
        for trial in range(10):
            print("...........", trial, "/ 10 ...........")
            if str(trial) in k_results:
                continue
            temp_path.mkdir()
            k_specific =shuffle(specific_train_df, n_samples=k)
            k_general = pd.concat([general_train_df, k_specific])

            model_general = train_model(
                k_general,
                normalizer_path=temp_path/"normalizer.pkl",
                organism_encoder_path=temp_path/"encoder.pkl",
                train_test_column="train",
                train_test_value="True",
                **MODEL_PARAMS
            )
            Xd, Xp, y_real = data_load(
                test_df,
                normalizer_path=temp_path/"normalizer.pkl",
                organism_encoder_path=temp_path/"encoder.pkl",
                train_test_column="train",
                train_test_value="False",
                **DATA_PARAMS
            )
            model_general.eval()
            with torch.no_grad():
                y_pred_general = model_general(Xd, Xp).detach().cpu().numpy()
            shutil.rmtree(temp_path)
            temp_path.mkdir()
            model_specific = train_model(
                k_specific,
                normalizer_path=temp_path/"normalizer.pkl",
                organism_encoder_path=temp_path/"encoder.pkl",
                train_test_column="train",
                train_test_value="True",
                **MODEL_PARAMS
                )
            Xd, Xp, y_real = data_load(
                test_df,
                normalizer_path=temp_path/"normalizer.pkl",
                organism_encoder_path=temp_path/"encoder.pkl",
                train_test_column="train",
                train_test_value="False",
                **DATA_PARAMS
            )
            model_specific.eval()
            with torch.no_grad():
                y_pred_specific = model_specific(Xd, Xp).detach().cpu().numpy()
            y_real = y_real.detach().cpu().numpy()
            shutil.rmtree(temp_path)
            k_results[trial] = dict(
                general = mean_squared_error(y_real, y_pred_general)**.5,
                specific = mean_squared_error(y_real, y_pred_specific)**.5,
            )
            k_results[trial]["difference"] = k_results[trial]["general"] - k_results[trial]["specific"]
            with open(JSON_PATH, "w") as file:
                json.dump(general_results, file, indent=1)

data = pd.read_csv("dataset/split_by_compound/compound_split_0.csv")
for species in data.target_organism.unique():
    print("===========", species.upper(), "===========")
    evaluate_specie(data, species)