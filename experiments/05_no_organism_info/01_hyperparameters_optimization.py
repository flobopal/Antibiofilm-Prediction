import pandas as pd
import sys
import os
sys.path.append(os.getcwd())
from model.train_and_optimize.optimizer import FeedForwardObjective, do_study

dataset = pd.read_csv("dataset/split_by_compound/compound_split_0.csv")
dataset = dataset[dataset["train"] == True].copy()

study = do_study(
    dataset,
    database="sqlite:///experiments/database.db",
    name="ablation no organism",
    n_trials=1000,
    objective=FeedForwardObjective,
    features_start=6,
    features_end=None,
    organism_column="target_organism",
    output_column="pIC50",
    normalize_features=True,
    normalizer_start=768,
    normalizer_end=900,
    metric="max_rmse",
    balance_organisms=True,
    include_organism_features=False,
    fold_column="kfold",
)