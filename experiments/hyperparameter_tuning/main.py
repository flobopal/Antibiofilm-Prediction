import sys
import os
from pathlib import Path
sys.path.append(os.getcwd())

from script.utils.data_load import data_load
from model.train_and_optimize.optimizer import do_study, Objective
folder = Path("experiments", "hyperparameter_tuning")


Xd, Xp, y = data_load("data/Antibiofilm data.csv", features_start=5, organism_encoder_path=folder / "encoder.pkl", organism_column='target_organism',
          normalizer_path=folder / "normalizer.pkl", output_column='pIC50', train_test_column='train', train_test_value='True', normalizer_start=1167, normalizer_end=2000)

study = do_study(Xd, Xp, y, "hiperparameters_optimization", 100, "sqlite:///experiments/database.db", Objective)
print(study.best_params)