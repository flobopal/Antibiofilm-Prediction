from typing import Callable

import optuna
import pandas as pd
import torch
import gc
from model.decoder import FeedForwardNetwork, FeedForwardNetworkParams
from model.full_model import FullModel
from model.interaction import MoleculeOrganismInteractionParams
from script.tasks.cross_validate import cross_validate
from script.utils.activation_functions import list_names

def clean():
    """
    Performs memory cleanup for both CPU and GPU.

    This function triggers Python's garbage collector to free up unreferenced memory.
    If a CUDA-capable GPU is available, it also clears the CUDA memory cache and
    collects inter-process communication (IPC) resources to help prevent memory leaks
    and fragmentation during intensive GPU computations.
    """
    gc.collect()
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def suggest_scheduler_kwargs(scheduler: str, num_epochs: int, trial: optuna.Trial) -> dict:
    """
    Suggests hyperparameters for different learning rate schedulers using Optuna.
    Args:
        scheduler (str): The type of scheduler to suggest parameters for. 
            Supported values are 'step', 'exponential', and 'cosine'.
        num_epochs (int): The total number of training epochs, used for certain scheduler parameters.
        trial (optuna.Trial): The Optuna trial object used to suggest hyperparameter values.
    Returns:
        dict: A dictionary containing suggested keyword arguments for the specified scheduler type.
            - For 'step': {'step_size': int, 'gamma': float}
            - For 'exponential': {'gamma': float}
            - For 'cosine': {'T_max': float, 'eta_min': float}
            - For unsupported schedulers: an empty dictionary.
    """
    
    if scheduler == 'step':
        return {
            "step_size": trial.suggest_int('schel_step_size', 5, 30, step=5),
            "gamma": trial.suggest_float('schel_gamma', 0.1, 0.9)
        }
    if scheduler == 'exponential':
        return {
            'gamma': trial.suggest_float('schel_gamma', 0.8, 0.99)
        }
    if scheduler == 'cosine':
        return {
            'T_max': num_epochs / 2**trial.suggest_int("schel_t_max_mod", 0,5),
            'eta_min' :  trial.suggest_categorical("shchel_eta_min", [0.0, 1e-6, 1e-5, 1e-4, 1e-3])
        }
    return {}


class Objective:
    def __init__(
            self,
            dataset: pd.DataFrame,
            metrics: str,
            features_start: int | str,
            output_column: str,
            fold_column: str = "kfold",
            organism_column: str | None = None,
            features_end: int | str | None = None,
            normalize_features: bool = True,
            normalizer_start: int | None = None,
            normalizer_end: int | None = None,
            balance_organism: bool = False,
            include_organism_features: bool = True,

    ):
        self.dataset = dataset
        self.metrics = metrics
        self.features_start = features_start
        self.features_end = features_end
        self.output_column = output_column
        self.fold_column = fold_column
        self.organism_column = organism_column
        self.normalize_features = normalize_features
        self.normalizer_start = normalizer_start
        self.normalizer_end = normalizer_end
        self.balance_organism = balance_organism
        self.include_organism_features = include_organism_features

    def model_factory(self, **model_kwargs) -> Callable[[int, int], torch.nn.Module]:
        pass

    def suggest_params(self, trial: optuna.Trial) -> dict:
        pass

    def cross_validate(self, model_params: dict) -> list[float]:
        values = cross_validate(
            self.dataset,
            self.model_factory(**model_params),
            self.features_start,
            self.output_column,
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs={"lr": model_params["lr"]},
            task_type="regression",
            fold_column=self.fold_column,
            features_end=self.features_end,
            normalize_features=self.normalize_features,
            normalizer_start=self.normalizer_start,
            normalizer_end=self.normalizer_end,
            metrics=self.metrics,
            include_organism_features=self.include_organism_features,
            balance_organism=self.balance_organism,
            num_epochs=model_params["num_epochs"],
            scheduler_name=model_params["scheduler_name"],
            scheduler_kwargs=model_params["scheduler_kwargs"],
        )
        return sum(values) / len(values)

    
    def __call__(self, trial: optuna.Trial) -> float:
        clean()
        params = self.suggest_params(trial)
        return self.cross_validate(params)

class FullModelObjective(Objective):

    def suggest_params(self, trial: optuna.Trial) -> dict:
        embed_dim = 2**trial.suggest_int("log2_embed_size", 2,7)
        num_layers = trial.suggest_int("num_layers", 1, 4)
        hidden_dims = []
        activations = []
        for layer_index in range(num_layers):
            hidden_dims.append(
                2**trial.suggest_int(f"log2_layer_{layer_index}", 2, 10)
            )
            activations.append(
                trial.suggest_categorical(
                    f"activation_{layer_index+1}",
                    list_names()
                )
            )
        activations.append(trial.suggest_categorical("final_activation", list_names()))
        if activations[-1] == 'elu':
            last_activation_params = {
                "alpha": trial.suggest_float("elu_alpha", 0.1, 2)
            }
        elif activations[-1] == 'leaky_relu':
            last_activation_params = {
                "negative_slope": trial.suggest_float("lRelu_slope", 1e-3, 0.3, log=True)
            }
        else:
            last_activation_params = None
        num_heads = 2**trial.suggest_int("log2_num_heads", 0, 4)
        pooling = trial.suggest_categorical("pooling", ['mean', 'max', 'linear'])
        dropout = trial.suggest_float('dropout', 0, 0.5)
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        num_epochs = 2**trial.suggest_int('epochs', 6, 10)
        scheduler_name = trial.suggest_categorical(
            "scheduler",
            ['step', 'exponential', 'cosine', 'none']
        )
        scheduler_kwargs = suggest_scheduler_kwargs(scheduler_name, num_epochs, trial)

        return dict(
            embed_dim=embed_dim,
            hidden_dims=hidden_dims,
            activations=activations,
            num_heads=num_heads,
            pooling=pooling,
            dropout=dropout,
            lr=lr,
            num_epochs=num_epochs,
            scheduler_name=scheduler_name,
            scheduler_kwargs=scheduler_kwargs,
            last_activation_params=last_activation_params
        )


    def model_factory(self, **model_kwargs) -> Callable[[int, int], FullModel]:
        def _model_factory(Xd_dim: int, Xo_dim: int) -> torch.nn.Module:
            interaction_params = MoleculeOrganismInteractionParams(
                Xd_dim=Xd_dim,
                Xo_dim=Xo_dim,
                embed_dim=model_kwargs["embed_dim"],
                num_heads=model_kwargs["num_heads"],
                pooling=model_kwargs["pooling"],
                dropout=model_kwargs["dropout"],
            )
            decoder_params = FeedForwardNetworkParams(
                input_dim=interaction_params.embed_dim,
                hidden_dims=model_kwargs["hidden_dims"],
                activations=model_kwargs["activations"],
                dropout=model_kwargs["dropout"],
                last_activation_params=model_kwargs["last_activation_params"]
            )
            return FullModel(
                Xd_dim=Xd_dim,
                Xo_dim=Xo_dim,
                interaction_params=interaction_params,
                decoder_params=decoder_params
            )
        return _model_factory

class FeedForwardObjective(Objective):
    
    def suggest_params(self, trial: optuna.Trial) -> dict:
        num_layers = trial.suggest_int("num_layers", 1, 5)
        hidden_dims = []
        activations = []
        for layer_index in range(num_layers):
            hidden_dims.append(
                2**trial.suggest_int(f"log2_layer_{layer_index}", 2, 12)
            )
            activations.append(
                trial.suggest_categorical(
                    f"activation_{layer_index+1}",
                    list_names()
                )
            )
        activations.append(trial.suggest_categorical("final_activation", list_names()))
        if activations[-1] == 'elu':
            last_activation_params = {
                "alpha": trial.suggest_float("elu_alpha", 0.1, 2)
            }
        elif activations[-1] == 'leaky_relu':
            last_activation_params = {
                "negative_slope": trial.suggest_float("lRelu_slope", 1e-3, 0.3, log=True)
            }
        else:
            last_activation_params = None
        dropout = trial.suggest_float('dropout', 0, 0.5)
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        num_epochs = 2**trial.suggest_int('epochs', 6, 10)
        scheduler_name = trial.suggest_categorical(
            "scheduler",
            ['step', 'exponential', 'cosine', 'none']
        )
        scheduler_kwargs = suggest_scheduler_kwargs(scheduler_name, num_epochs, trial)

        return dict(
            hidden_dims=hidden_dims,
            activations=activations,
            dropout=dropout,
            lr=lr,
            num_epochs=num_epochs,
            scheduler_name=scheduler_name,
            scheduler_kwargs=scheduler_kwargs,
            last_activation_params=last_activation_params
        )

    def model_factory(self, **model_kwargs) -> Callable[[int, int], FeedForwardNetwork]:

        def _model_factory(Xd_dim: int, Xo_dim: int) -> torch.nn.Module:
            params = FeedForwardNetworkParams(
                input_dim=Xd_dim + Xo_dim,
                hidden_dims=model_kwargs["hidden_dims"],
                activations=model_kwargs["activations"],
                dropout=model_kwargs["dropout"],
                last_activation_params=model_kwargs["last_activation_params"]
            )
            return FeedForwardNetwork.from_params(params)
        return _model_factory


def do_study(
        dataset: pd.DataFrame,
        database: str,
        name: str,
        n_trials: int = 50,
        objective=Objective,
        features_start: int | str = 6,
        features_end: int | str | None = None,
        organism_column: str = "target_organism",
        output_column: str = "pIC50",
        normalize_features: bool = True,
        normalizer_start: int | None = 768,
        normalizer_end: int | None = None,
        metric: str = "r2",
        balance_organisms: bool = True,
        include_organism_features: bool = True,
        fold_column: str | None = None,
        ):

    direction = "maximize" if metric in ["r2"] else "minimize"


    study = optuna.create_study(
        direction=direction,
        study_name=name,
        storage=database,
        load_if_exists=True,
    )

    study.optimize(
        objective(
            dataset,
            metric,
            features_start,
            output_column,
            fold_column=fold_column,
            organism_column=organism_column,
            features_end=features_end,
            normalize_features=normalize_features,
            normalizer_start=normalizer_start,
            normalizer_end=normalizer_end,
            balance_organism=balance_organisms,
            include_organism_features=include_organism_features
        ),
        n_trials=n_trials,
    )
    return study
