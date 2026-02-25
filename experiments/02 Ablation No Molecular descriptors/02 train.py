import sys
import os
from pathlib import Path
sys.path.append(os.getcwd())

from script.utils.data_load import data_load
from model.train_and_optimize.trainer import Trainer

folder = Path("experiments", "02 Ablation No Molecular descriptors")

Xd, Xp, y = data_load("experiments/Antibiofilm MF+MD.csv", features_start=5, features_end=773, organism_encoder_path=folder / "encoder.pkl", organism_column='target_organism',
         output_column='pIC50', train_test_column='train', train_test_value='True')

model = Trainer(
    Xd, Xp, y,
    embed_dim=2**4,
    hidden_dims=[2**4, 2**5, 2**10],
    activations=["relu", "leaky_relu", "leaky_relu", "leaky_relu"],
    num_heads=2**2,
    pooling="max",
    dropout=0.023,
    lr=4.14e-4,
    num_epochs=2**9,
    scheduler_name="exponential",
    scheduler_kwargs={"gamma": 0.887}
)

model_path = Path(os.path.abspath(__file__)).parent / "antibiofilm_model.pth"

model.train(model_path)