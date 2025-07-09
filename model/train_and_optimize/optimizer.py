import optuna
import torch
import gc
from model.train_and_optimize.trainer import Trainer
from script.utils.activation_functions import list_names

def clean():
    gc.collect()
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def suggest_scheduler_kwargs(scheduler: str, num_epochs: int, trial: optuna.Trial) -> dict:
    
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
    def __init__(self, Xd, Xp, y):
        self.Xd = Xd
        self.Xp = Xp
        self.y = y
    def __call__(self, trial: optuna.Trial) -> float:
        # Tuneable params
        embed_dim = 2**trial.suggest_int("log2_embed_size", 2,7)
        num_layers = trial.suggest_int("num_layers", 1, 4)
        hidden_dims = []
        activations = [trial.suggest_categorical('activation_0', list_names())]
        for layer_index in range(num_layers):
            hidden_dims.append(
                2**trial.suggest_int(f"log2_layer_{layer_index}", 2, 7)
            )
            activations.append(
                trial.suggest_categorical(
                    f"activation_{layer_index+1}",
                    list_names()
                )
            )
        num_heads = 2**trial.suggest_int("log2_num_heads", 0, 6)
        pooling = trial.suggest_categorical("pooling", ['mean', 'max', 'linear'])
        dropout = trial.suggest_float('dropout', 0, 0.15)
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        num_epochs = 2**trial.suggest_int('epochs', 6, 10)
        scheduler_name = trial.suggest_categorical(
            "scheduler",
            ['step', 'exponential', 'cosine', 'none']
        )
        scheduler_kwargs = suggest_scheduler_kwargs(scheduler_name, num_epochs, trial)

        # Create model
        model = Trainer(
            Xd=self.Xd,
            Xp=self.Xp,
            y=self.y,
            embed_dim=embed_dim,
            hidden_dims=hidden_dims,
            activations=activations,
            dropout=dropout,
            num_heads=num_heads,
            pooling=pooling,
            num_epochs=num_epochs,
            lr=lr,
            scheduler_name=scheduler_name,
            scheduler_kwargs=scheduler_kwargs)
        
        # Cross validations
        vals = model.cross_validation(3)
        return sum(vals) / 3


def do_study(Xd, Xp, y, n_trials=50):
    clean()
    study = optuna.create_study(direction="minimize")
    study.optimize(Objective(Xd, Xp, y), n_trials=n_trials)
    return study