import pandas as pd
import torch
import sys
import os
from pathlib import Path
sys.path.append(os.getcwd())
from model.full_model import train_model
from script.utils.data_load import data_load

results = pd.DataFrame(columns=["split_number", "target_organism", "curated_smiles", "real_pIC50", "pred_pIC50"])
current_path = Path(os.path.abspath(__file__)).parent
if __name__ == "__main__":
    for partition in range(10):
        print("=========== SPLIT",partition,"===============")
        datapath = f"dataset/split_by_compound/compound_split_{partition}.csv"
        model_path = current_path / f"split_{partition}"
        model_path.mkdir(exist_ok=True)
        model = train_model(
                data_path=datapath,
                features_start=774,
                features_end=None,
                organism_encoder_path=model_path / "encoder.pkl",
                organism_column="target_organism",
                output_column="pIC50",
                normalizer_path=model_path / "normalizer.pkl",
                normalizer_start=0,
                normalizer_end=132,
                train_test_column="train",
                train_test_value="True",
                weighted_loss=True,
                embed_dim=2**8,
                hidden_dims=[2**7],
                activations=["relu", "identity"],
                # last_activation_params={},
                dropout= 0.1232787512025156,
                lr=0.00826060533223491,
                num_epochs=2**10,
                batch_size=64,
                scheduler_name="exponential",
                scheduler_kwargs={"gamma": 0.9191111257696735},
                save_path=model_path / "antibiofilm_model.pth",
            )

        Xd, Xp, y_real = data_load(
            data_path=datapath,
            features_start=774,
            features_end=None,
            organism_encoder_path=model_path / "encoder.pkl",
            organism_column="target_organism",
            output_column="pIC50",
            normalizer_path=model_path / "normalizer.pkl",
            normalizer_start=0,
            normalizer_end=132,
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