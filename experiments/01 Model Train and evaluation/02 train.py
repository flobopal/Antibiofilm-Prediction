import sys
import os
from pathlib import Path
sys.path.append(os.getcwd())

from script.utils.data_load import data_load
from model.train_and_optimize.trainer import Trainer

folder = Path("experiments", "01 Model Train and evaluation")

Xd, Xp, y = data_load("experiments/Antibiofilm MF+MD.csv", features_start=5, organism_encoder_path=folder / "encoder.pkl", organism_column='target_organism',
          normalizer_path=folder / "normalizer.pkl", output_column='pIC50', train_test_column='train', train_test_value='True', normalizer_start=768, normalizer_end=2000)

model = Trainer(
    Xd, Xp, y,
    embed_dim=2**4,
    hidden_dims=[2**5, 2**9],
    activations=["tanh", "tanh", "leaky_relu"],
    num_heads=2**2,
    pooling="max",
    dropout=0.0311,
    lr=9.27e-5,
    num_epochs=2**8,
    scheduler_name="cosine",
    scheduler_kwargs={"T_max": 2**8, "eta_min": 0.0001}
)

model_path = Path(os.path.abspath(__file__)).parent / "antibiofilm_model.pth"

model.train(model_path)