import sys
import os
from pathlib import Path
sys.path.append(os.getcwd())

from script.utils.data_load import data_load
from model.train_and_optimize.trainer import Trainer

folder = Path("experiments", "03 Ablation No Molformer")

Xd, Xp, y = data_load("experiments/Antibiofilm MF+MD.csv", features_start=773, organism_encoder_path=folder / "encoder.pkl", organism_column='target_organism',
        normalizer_path=folder / "normalizer.pkl", output_column='pIC50', train_test_column='train', train_test_value='True')

model = Trainer(
    Xd, Xp, y,
    embed_dim=2**4,
    hidden_dims=[2**3, 2**6, 2**3],
    activations=["gelu", "leaky_relu", "tanh", "leaky_relu"],
    num_heads=2**3,
    pooling="max",
    dropout=0.07459,
    lr=0.00014,
    num_epochs=2**7,
    scheduler_name="cosine",
    scheduler_kwargs={"T_max": 2**5, "eta_min": 1e-6}
)

model_path = Path(os.path.abspath(__file__)).parent / "antibiofilm_model.pth"

model.train(model_path)