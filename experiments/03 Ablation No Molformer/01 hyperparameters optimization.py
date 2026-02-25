import sys
import os
from pathlib import Path
sys.path.append(os.getcwd())

from script.utils.data_load import data_load
from model.train_and_optimize.optimizer import do_study, Objective
folder = Path("experiments", "03 Ablation No Molformer")

Xd, Xp, y = data_load("experiments/Antibiofilm MF+MD.csv", features_start=773, organism_encoder_path=folder / "encoder.pkl", organism_column='target_organism',
        normalizer_path=folder / "normalizer.pkl", output_column='pIC50', train_test_column='train', train_test_value='True')


study = do_study(Xd, Xp, y, "Ablation no Molformer", 100, "sqlite:///experiments/database.db", Objective)
print(study.best_params)