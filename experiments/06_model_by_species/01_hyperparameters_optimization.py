import pandas as pd
import sys
import os
sys.path.append(os.getcwd())
from model.train_and_optimize.optimizer import FeedForwardObjective, do_study

dataset = pd.read_csv("dataset/split_by_compound/compound_split_0.csv")
dataset = dataset[dataset["train"] == True].copy()

for species in dataset.target_organism.unique():
    subdata = dataset[dataset.target_organism.eq(species)]
    study = do_study(
        subdata,
        database="sqlite:///experiments/database.db",
        name=f"species specific {species}",
        n_trials=200,
        objective=FeedForwardObjective,
        features_start=6,
        features_end=None,
        organism_column="target_organism",
        output_column="pIC50",
        normalize_features=True,
        normalizer_start=768,
        normalizer_end=900,
        metric="r2",
        balance_organisms=True,
        include_organism_features=False,
        fold_column="kfold",
    )