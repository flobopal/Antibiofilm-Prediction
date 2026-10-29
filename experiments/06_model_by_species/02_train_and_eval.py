import pandas as pd
import torch
import sys
import os
from pathlib import Path
import optuna
from tqdm import tqdm
sys.path.append(os.getcwd())
from model.decoder import train_mlp
from script.utils.data_load import data_load

results = pd.DataFrame(columns=["split_number", "target_organism", "curated_smiles", "real_pIC50", "pred_pIC50"])
current_path = Path(os.path.abspath(__file__)).parent
database = "sqlite:///" + str(current_path.parent / "database.db")

for study_name in optuna.get_all_study_names(database):
    if not study_name.startswith("species specific"):
        continue
    species = study_name.split(" ", 2)[2]
    print("======", species.upper(), "=========")
    (current_path / species).mkdir(exist_ok=True)

    params = optuna.load_study(study_name=study_name, storage=database).best_params
    hidden_dims = [2**value for key, value in params.items() if key.startswith("log2_layer_")]
    activations = [value for key, value in params.items() if key.startswith("activation_")]
    activations.append(params["final_activation"])
    dropout = params["dropout"]
    lr = params["lr"]
    epochs = 2**params["epochs"]
    scheduler = params["scheduler"]
    scheduler_params = None
    if scheduler == "none":
        scheduler = None
    elif scheduler == "step":
        scheduler_params = {"gamma": params["schel_gamma"], "step_size": params["schel_step_size"]}
    elif scheduler == "exponential":
        scheduler_params = {"gamma": params["schel_gamma"]}
    elif scheduler == "cosine":
        scheduler_params = {"T_max": epochs//2**params["schel_t_max_mod"], "eta_min": params["shchel_eta_min"]}
    last_activation_params = None
    if "elu_alpha" in params:
        last_activation_params = {"alpha": params["elu_alpha"]}
    if "lRelu_slope" in params:
        last_activation_params = {"negative_slope": params["lRelu_slope"]}

    for split_number in range(10):
        model_path = current_path / species / f"split_{split_number}"
        model_path.mkdir(exist_ok=True)
        datafile = f"dataset/split_by_compound/compound_split_{split_number}.csv"
        model = train_mlp(
            data_path=datafile,
            features_start=6,
            normalizer_path=model_path / "normalizer.pkl",
            normalizer_start=768,
            normalizer_end=900,
            output_column="pIC50",
            organism_column="target_organism",
            organism_filter=species,
            use_organisms_features=False,
            train_test_column="train",
            train_test_value="True",
            weighted_loss=True,
            hidden_dims=hidden_dims,
            activations=activations,
            dropout=dropout,
            lr=lr,
            num_epochs=epochs,
            scheduler_name=scheduler,
            scheduler_kwargs=scheduler_params,
            last_activation_params=last_activation_params,
            save_path=model_path / "antibiofilm_model.pth"
        )

        Xd, Xp, y_real = data_load(
                    data_path=datafile,
                    features_start=6,
                    features_end=None,
                    organism_encoder_path=model_path / "encoder.pkl",
                    organism_column="target_organism",
                    organism_filter=species,
                    output_column="pIC50",
                    normalizer_path=model_path / "normalizer.pkl",
                    normalizer_start=768,
                    normalizer_end=900,
                    train_test_column="train",
                    train_test_value="False",
                )
        model.eval()
        with torch.no_grad():
            y_pred = model(Xd).detach().cpu().numpy()

        eval_df = pd.read_csv(datafile, index_col=0).iloc[:,:5]
        eval_df = eval_df[~eval_df.train & eval_df.target_organism.eq(species)].copy()
        eval_df["pred_pIC50"] = y_pred
        eval_df["split_number"] = split_number
        eval_df.rename(columns={"pIC50": "real_pIC50"}, inplace=True)

        results = pd.concat([results, eval_df[["split_number", "target_organism", "curated_smiles", "real_pIC50", "pred_pIC50"]]])

results.to_csv(current_path / "predictions.csv")