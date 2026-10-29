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
                features_start=6,
                features_end=774,
                organism_encoder_path=model_path / "encoder.pkl",
                organism_column="target_organism",
                output_column="pIC50",
                train_test_column="train",
                train_test_value="True",
                embed_dim=2**9,
                hidden_dims=[2**7, 2**4],
                activations=["leaky_relu", "sigmoid", "relu"],
                # last_activation_params={},
                dropout= 0.4906741537826445,
                lr=0.004279464425730061,
                num_epochs=2**9,
                batch_size=64,
                scheduler_name=None,
                scheduler_kwargs=None,#{"step_size":5, "gamma": 0.7299717892570496},
                weighted_loss=True,
                save_path=model_path / "antibiofilm_model.pth",
            )

        Xd, Xp, y_real = data_load(
            data_path=datapath,
            features_start=6,
            features_end=774,
            organism_encoder_path=model_path / "encoder.pkl",
            organism_column="target_organism",
            output_column="pIC50",
            train_test_column="train",
            train_test_value="False",
        )
        model.eval()
        Xd.to
        with torch.no_grad():
            y_pred = model(Xd, Xp).detach().cpu().numpy()

        eval_df = pd.read_csv(datapath, index_col=0).iloc[:,:5]
        eval_df = eval_df[~eval_df.train]
        eval_df["pred_pIC50"] = y_pred
        eval_df["split_number"] = partition
        eval_df.rename(columns={"pIC50": "real_pIC50"}, inplace=True)

        results = pd.concat([results, eval_df[["split_number", "target_organism", "curated_smiles", "real_pIC50", "pred_pIC50"]]])

    results.to_csv(current_path / "predictions.csv")