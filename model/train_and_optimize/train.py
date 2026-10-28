import sys
import os
from pathlib import Path
sys.path.append(os.getcwd())
from model.full_model import train_model


BEST_PARAMS = dict(
    embed_dim=2**10,
    hidden_dims=[2**5,2**4,2**3],
    activations=["leaky_relu", "tanh", "tanh", "elu"],
    # last_activation_params={"negative_slope": 0.0016057464974335671},
    dropout= 0.28105987322261655,
    lr=5.270786745050085e-5,
    num_epochs=2**10,
    batch_size=64,
    scheduler_name="cosine",
    scheduler_kwargs={"T_max": 2**6, "eta_min": 0.001},
    weighted_loss=True
)
if __name__ == "__main__":

    datapath = Path("dataset/Antibiofilm data.csv")
    model_path = Path("antibiofilm checkpoint")
    model_path.mkdir(exist_ok=True)
    model = train_model(
            data_path=datapath,
            features_start=5, # No fold column in this dataset
            features_end=None,
            organism_encoder_path=model_path / "encoder.pkl",
            organism_column="target_organism",
            output_column="pIC50",
            normalizer_path=model_path / "normalizer.pkl",
            normalizer_start=768,
            normalizer_end=900,
            save_path=model_path / "antibiofilm_model.pth",
            **BEST_PARAMS
        )