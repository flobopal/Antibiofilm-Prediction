import sys
import os
sys.path.append(os.getcwd())

from pathlib import Path
import numpy as np
import optuna

from sklearn.model_selection import KFold

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, Subset

from script.utils.data_load import data_load
from script.utils.activation_functions import list_names
from script.utils.metrics import evaluate

from model.decoder import FeedForwardNetwork, FeedForwardNetworkParams
from model.train_and_optimize.optimizer import suggest_scheduler_kwargs, clean

from script.tasks.train import train_model


class Objective:
    def __init__(self, X, y, metrics):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
        self.metrics = metrics

    def __call__(self, trial: optuna.Trial):
        clean()

        #Tuneable params
        num_layers = trial.suggest_int("num_layers", 1, 5)
        hidden_dims = []
        activations = []
        for layer_index in range(num_layers):
            hidden_dims.append(
                2**trial.suggest_int(f"log2_layer_{layer_index}", 2, 14)
            )
            activations.append(
                trial.suggest_categorical(
                    f"activation_{layer_index+1}",
                    list_names()
                )
            )
        activations.append(trial.suggest_categorical("final_activation", list_names()))
        
        net_params = FeedForwardNetworkParams(
            input_dims=self.X.shape[1],
            hidden_dims=hidden_dims,
            activations=activations,
            dropout=trial.suggest_float("dropout", 0, 0.15)
        )
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        num_epochs = 2**trial.suggest_int('epochs', 6, 10)
        scheduler_name = trial.suggest_categorical(
            "scheduler",
            ['step', 'exponential', 'cosine', 'none']
        )
        scheduler_kwargs = suggest_scheduler_kwargs(scheduler_name, num_epochs, trial)

        #Cross validation
        K_FOLDS = 3
        VERBOSE = True
        BATCH_SIZE = 64
        print(self.X, self.y)
        dataset = TensorDataset(self.X, self.y)
        kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
        val_losses = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            clean()
            if VERBOSE:
                print(f"\n--- Fold {fold + 1}/{K_FOLDS} ---")

            # Dataloaders
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

            # Initialize new model
            model = FeedForwardNetwork.from_params(net_params)
            optimizer = Adam(model.parameters(), lr=lr)

            # Training
            train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                task_type="regression",
                num_epochs=num_epochs,
                scheduler_name=scheduler_name,
                scheduler_kwargs=scheduler_kwargs,
                verbose=VERBOSE
            )

            # Final fold validation
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.eval()
            X_test, y_test = map(torch.cat,zip(*list(val_loader)))
            y_pred = model(X_test.to(device))
            val_losses.append(evaluate("r2", y_test, y_pred))

        return sum(val_losses)/K_FOLDS

def do_study(X, y, name, n_trials, database):
    study = optuna.create_study(
        direction="maximize",
        study_name=name,
        storage=database,
        load_if_exists=True
    )
    study.optimize(Objective(X, y, "r2"), n_trials=n_trials)
    return study

if __name__ == "__main__":
    folder = Path("experiments", "05 No context")

    X, _, y = data_load("experiments/Antibiofilm MF+MD.csv", features_start=5, organism_encoder_path=folder / "encoder.pkl", organism_column='target_organism',
            normalizer_path=folder / "normalizer.pkl", output_column='pIC50', train_test_column='train', train_test_value='True', normalizer_start=768, normalizer_end=2000)

    study = do_study(X, y, "Ablation no context higher dimension limit", 100, "sqlite:///experiments/database.db")
    print(study.best_params)

