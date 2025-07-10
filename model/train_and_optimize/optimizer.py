import optuna
import torch
import gc
from model.train_and_optimize.trainer import Trainer
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
    """
    Objective class for hyperparameter optimization using Optuna.
    This class encapsulates the objective function to be minimized during
    hyperparameter search. It defines the search space for various model parameters,
    constructs a model with the suggested parameters, and evaluates its performance
    using cross-validation.
    Args:
        Xd (np.ndarray or torch.Tensor): Feature matrix for first modality.
        Xp (np.ndarray or torch.Tensor): Feature matrix for second modality.
        y (np.ndarray or torch.Tensor): Target labels.
    Methods:
        __call__(trial: optuna.Trial) -> float:
            Suggests hyperparameters, builds and trains the model, and returns the
            average cross-validation score for the given trial.
            Hyperparameters tuned:
                - embed_dim (int): Embedding dimension (power of 2 between 4 and 128).
                - num_layers (int): Number of hidden layers (1 to 4).
                - hidden_dims (List[int]): List of hidden layer sizes (powers of 2).
                - activations (List[str]): List of activation functions per layer.
                - num_heads (int): Number of attention heads (power of 2 between 1 and 64).
                - pooling (str): Pooling strategy ('mean', 'max', or 'linear').
                - dropout (float): Dropout rate (0 to 0.15).
                - lr (float): Learning rate (log-uniform between 1e-5 and 1e-2).
                - num_epochs (int): Number of training epochs (power of 2 between 64 and 1024).
                - scheduler_name (str): Learning rate scheduler type.
                - scheduler_kwargs (dict): Additional scheduler parameters.
            Returns:
                float: Average cross-validation score (mean of 3 folds).
    """
    def __init__(self, Xd, Xp, y):
        self.Xd = Xd
        self.Xp = Xp
        self.y = y
    def __call__(self, trial: optuna.Trial) -> float:
        clean()
        # Tuneable params
        embed_dim = 2**trial.suggest_int("log2_embed_size", 2,7)
        num_layers = trial.suggest_int("num_layers", 1, 3)
        hidden_dims = []
        activations = []
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
        activations.append(trial.suggest_categorical("final_activation", list_names()))
        num_heads = 2**trial.suggest_int("log2_num_heads", 0, 4)
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


def do_study(Xd, Xp, y, name: str= "model_optimization", n_trials=50, database: str=None):
    """
    Runs an Optuna hyperparameter optimization study for a given dataset and target.

    Parameters:
        Xd (array-like): Feature matrix for descriptors (e.g., embeddings or numerical features).
        Xp (array-like): Feature matrix for additional properties or related features.
        y (array-like): Target vector containing labels or values to predict.
        name (str, optional): Name of the study and default SQLite database filename. Defaults to "model_optimization".
        n_trials (int, optional): Number of optimization trials to run. Defaults to 50.
        database (str, optional): URI of the database to store the study results. 
                                  If None, a local SQLite file `{name}.db` is created and used.

    Returns:
        optuna.study.Study: The completed Optuna study object containing optimization results.

    Notes:
        - The `Objective` class or callable must be defined elsewhere and accept `(Xd, Xp, y)` as input to create the Optuna objective function.
        - If the specified database already exists, the study will be loaded and new trials appended.
        - The SQLite database file will be created automatically if it does not exist.
    """
    if database is None:
        database = f"sqlite:///{name}.db"

    study = optuna.create_study(
        direction="minimize",
        study_name=name,
        storage=database,
        load_if_exists=True)
    study.optimize(Objective(Xd, Xp, y), n_trials=n_trials)
    return study