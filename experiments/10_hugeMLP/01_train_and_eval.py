import pandas as pd
import torch
import sys
import os
from pathlib import Path
sys.path.append(os.getcwd())
from model.decoder import train_mlp
from script.utils.data_load import data_load

results = pd.DataFrame(columns=["split_number", "target_organism", "curated_smiles", "real_pIC50", "pred_pIC50"])
current_path = Path(os.path.abspath(__file__)).parent
if __name__ == "__main__":
    for partition in range(10):
        print("=========== SPLIT",partition,"===============")
        datapath = f"dataset/split_by_compound/compound_split_{partition}.csv"
        model_path = current_path / f"split_{partition}"
        model_path.mkdir(exist_ok=True)
        model = train_mlp(
                data_path=datapath,
                features_start=6,
                features_end=None,
                organism_encoder_path=model_path / "encoder.pkl",
                organism_column="target_organism",
                output_column="pIC50",
                normalizer_path=model_path / "normalizer.pkl",
                normalizer_start=768,
                normalizer_end=900,
                train_test_column="train",
                train_test_value="True",
                weighted_loss=True,
                hidden_dims=[2**11, 2**5, 2**6, 2**3],
                activations=[torch.nn.SiLU(), "leaky_relu", "tanh", "tanh", "elu"],
                # last_activation_params={},
                dropout= 0.28105987322261655,
                lr=5.270786745050085e-5,
                num_epochs=2**10,
                batch_size=64,
                scheduler_name="cosine",
                scheduler_kwargs={"T_max": 2**6, "eta_min": 0.001},
                save_path=model_path / "antibiofilm_model.pth",
            )

        Xd, Xp, y_real = data_load(
            data_path=datapath,
            features_start=6,
            features_end=None,
            organism_encoder_path=model_path / "encoder.pkl",
            organism_column="target_organism",
            output_column="pIC50",
            normalizer_path=model_path / "normalizer.pkl",
            normalizer_start=768,
            normalizer_end=900,
            train_test_column="train",
            train_test_value="False",
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

    results.to_csv(current_path / "predictions.csv")