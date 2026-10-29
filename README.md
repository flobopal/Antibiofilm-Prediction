# Antibiofilm Prediction

![License](https://img.shields.io/badge/license-MIT-green)

Code, datasets, and experiments for **“Species-Context Aware Quantitative Prediction of the Antibiofilm Activity of Small Molecules.”**

The model predicts small-molecule antibiofilm activity (`pIC50`) using molecular features and bacterial species identity.

## Model

The current implementation combines:

- **768 MolFormer embeddings** computed from SMILES.
- **217 RDKit descriptors** in the bundled datasets, giving 985 molecular features in total.
- **One-hot organism encoding**, with eight species in the bundled checkpoint.
- **Organism-conditioned FiLM** (feature-wise linear modulation) in [model/interaction.py](model/interaction.py).
- A configurable **feedforward regression network** in [model/decoder.py](model/decoder.py).

The interaction module projects molecular features through a linear layer, layer normalization, and SiLU. An organism projection produces `gamma` and `beta`; the representation passed to the decoder is:

```text
dropout((1 + tanh(gamma)) * molecular_representation + beta)
```

The organism projection is initialized to zero. [FullModel](model/full_model.py) combines this interaction module with the decoder and provides checkpoint saving and loading.

## Repository layout

| Path | Contents |
| --- | --- |
| `main.py` | CLI for organism combinations, embeddings, descriptors, and prediction |
| `model/` | FiLM interaction, regression decoder, full model, and training/optimization entry points |
| `script/tasks/` | Training loops and cross-validation |
| `script/utils/` | Data loading, normalization, organism encoding, metrics, weights, and split utilities |
| `script/embeddings/MolFormer/` | MolFormer feature extraction code and configuration |
| `antibiofilm checkpoint/` | Bundled model, organism encoder, and normalizer |
| `dataset/` | Prepared datasets, compound/scaffold splits, and dataset analysis |
| `experiments/` | Optimization, evaluation, ablations, baselines, and analysis notebooks |
| `test/` | Automated tests, including some tests for older APIs |

Run commands from the repository root; scripts use paths relative to this directory.

## Installation

```bash
git clone https://github.com/flobopal/Antibiofilm-Prediction.git
cd Antibiofilm-Prediction
conda env create -f environment.yml
conda activate antibiofilm
```

[environment.yml](environment.yml) records the original environment pins, including Python 3.13.3, PyTorch 2.5.1, RDKit 2025.03.2, scikit-learn 1.6.1, and Optuna 4.4.0. These pins and channels may need adjustment for your platform; the file is not a portable lockfile. [requirements.txt](requirements.txt) also contains the Conda package name `pytorch`; the corresponding pip package is `torch`.

The main code uses Python 3.11+ features and depends on PyTorch, NumPy, pandas, scikit-learn, joblib, tqdm, RDKit, and Optuna. Use scikit-learn 1.6.1 with the bundled preprocessing objects and RDKit 2025.03.2 when reproducing the descriptor set. The XGBoost baseline additionally needs `xgboost`, which is not listed in the environment files.

### MolFormer environment and weights

Skip this setup if you use the prepared feature CSVs. To generate embeddings from new SMILES, create a separate environment following [IBM's MolFormer environment instructions](https://github.com/IBM/molformer/blob/main/environment.md). The examples below call this environment `MolTran_CUDA11`.

Obtain the pretrained weights through the [upstream MolFormer repository](https://github.com/IBM/molformer). The local extractor expects this exact layout and capitalization:

```text
script/embeddings/MolFormer/Pretrained MoLFormer/
├── hparams.yaml
└── checkpoints/
    └── N-Step-Checkpoint_3_30000.ckpt
```

The configuration is included in this repository. Check that the checkpoint is present before running `embeddings`.

## Predict with the bundled model

The model and its preprocessing files are stored together:

```text
antibiofilm checkpoint/
├── antibiofilm_model.pth
├── encoder.pkl
└── normalizer.pkl
```

The directory name contains a space. Use these three files as a matched set, with the same feature order used during training.

### Python example: prepared data on CPU or GPU

Run this example from the repository root in the `antibiofilm` environment. It uses [dataset/Example data.csv](<dataset/Example data.csv>) and writes `predictions.csv`:

```python
from pathlib import Path

import pandas as pd
import torch

from model.full_model import FullModel
from script.utils.data_load import data_load

checkpoint_dir = Path("antibiofilm checkpoint")
data = pd.read_csv("dataset/Example data.csv")

Xd, Xp, _ = data_load(
    data,
    features_start=3,
    organism_column="target_organism",
    organism_encoder_path=checkpoint_dir / "encoder.pkl",
    normalizer_path=checkpoint_dir / "normalizer.pkl",
    normalizer_start=768,
    normalizer_end=900,
)

model = FullModel.load(checkpoint_dir / "antibiofilm_model.pth")
model = model.to(Xd.device).eval()

with torch.no_grad():
    predictions = model(Xd, Xp).detach().cpu().numpy()

output = data.iloc[:, :3].copy()
output["prediction"] = predictions
output.to_csv("predictions.csv", index=False)
```

`data_load` selects CUDA when available. `FullModel.load(path)` loads the checkpoint on CPU, including checkpoints saved on CUDA; move the model to `Xd.device` before inference, as shown above.

### Prepare features from SMILES

Start with a CSV such as `compounds.csv` containing a `smiles` column:

```csv
smiles
CCO
CC(=O)O
```

Generate all unique SMILES–organism combinations, MolFormer embeddings, and RDKit descriptors:

```bash
conda activate antibiofilm
python main.py organisms compounds.csv organisms.csv smiles

conda activate MolTran_CUDA11
python main.py embeddings organisms.csv embeddings.csv target_organism smiles

conda activate antibiofilm
python main.py descriptors embeddings.csv smiles descriptors.csv
```

`organisms` reads species from the bundled encoder by default and writes columns named `smiles` and `target_organism`. To select species explicitly:

```bash
python main.py organisms compounds.csv organisms.csv smiles \
    --organisms "Streptococcus mutans" "Staphylococcus aureus"
```

You can also set `--organism_encoder_path "path/to/encoder.pkl"`. If your input already contains SMILES–organism pairs, start with `embeddings` and supply your column names. The extractor retains only those two metadata columns and the embeddings; it does not preserve activity labels or split assignments.

RDKit descriptors are appended to the input columns, and the descriptor CSV includes a leading index column. The resulting layout is:

| Zero-based CSV positions | Contents |
| --- | --- |
| `0` | Saved dataframe index |
| `1` | Organism |
| `2` | SMILES |
| `3`–`770` | 768 MolFormer embeddings, named `0` through `767` |
| `771` onward | RDKit descriptors |

The descriptor count and order depend on RDKit's version. For the bundled model, match the 217 descriptor columns in `dataset/Example data.csv`. To use the Python prediction example with new compounds, read `descriptors.csv` instead.

### CLI prediction

The CLI loads the bundled checkpoint on CPU, moves the model to the input tensors' device (CUDA when available), and transfers predictions to CPU for CSV export:

```bash
python main.py prediction \
    "dataset/Example data.csv" target_organism predictions.csv
```

Replace the input with `descriptors.csv` for newly generated features. The output contains the columns before `features_start`, followed by `prediction`. A subcommand is required; use `python main.py --help` to list them.

| Prediction option | CLI default | Meaning |
| --- | --- | --- |
| `--features_start` | `3` | First feature column in the CSV |
| `--features_end` | End of CSV | Exclusive end of the feature columns |
| `--normalizer_start` | `768` | First feature to normalize, relative to the selected feature matrix |
| `--normalizer_end` | `900` | Exclusive end of normalization, matching the bundled normalizer |
| `--organism_encoder_path` | `antibiofilm checkpoint/encoder.pkl` | Fitted organism encoder |
| `--normalizer_path` | `antibiofilm checkpoint/normalizer.pkl` | Fitted feature scaler |
| `--model_checkpoint_path` | `antibiofilm checkpoint/antibiofilm_model.pth` | Model checkpoint |

The supplied scaler was fitted on feature positions `[768:900]`, covering 132 descriptors. Remaining features are still model inputs but are not scaled. All CLI options use underscores. See `python main.py prediction --help` for the parser's options.

## Training and data conventions

Prepared training files include `target_organism`, `curated_smiles`, `pIC50`, a `train` flag, and molecular features. Split files also include `kfold` assignments for cross-validation.

| Dataset | First feature position | Feature count |
| --- | --- | --- |
| `dataset/Example data.csv` | `3` | `985` |
| `dataset/Antibiofilm data.csv` | `5` | `985` |
| `dataset/split_by_compound/compound_split_*.csv` | `6` | `985` |
| `dataset/split_by_scaffold/scaffold_split_*.csv` | `6` | `985` |

These positions assume `pd.read_csv(path)` without `index_col`. Integer feature ranges use zero-based positions with an exclusive end. Normalization indices always refer to the selected feature matrix, rather than the full CSV.

`data_load` loads an existing scaler/encoder or fits and saves one when its path does not exist. Create output directories first. Fit preprocessing on training rows, then reuse those files for evaluation and inference. Use a fresh artifact directory for each independent training run; existing files are reused. Unknown species are rejected by the bundled encoder.

### Train the full model

The current training API is `model.full_model.train_model`. This example trains on one compound split and saves a separate set of artifacts:

```python
from pathlib import Path

from model.full_model import train_model
from model.train_and_optimize.train import BEST_PARAMS

output_dir = Path("runs/compound_split_0")
output_dir.mkdir(parents=True, exist_ok=True)

model = train_model(
    data_path="dataset/split_by_compound/compound_split_0.csv",
    features_start=6,
    organism_column="target_organism",
    output_column="pIC50",
    train_test_column="train",
    train_test_value="True",
    organism_encoder_path=output_dir / "encoder.pkl",
    normalizer_path=output_dir / "normalizer.pkl",
    normalizer_start=768,
    normalizer_end=900,
    save_path=output_dir / "antibiofilm_model.pth",
    **BEST_PARAMS,
)
```

`BEST_PARAMS` specifies a 1024-dimensional interaction representation, decoder widths `[32, 16, 8]`, batch size 64, 1024 epochs, Adam, a cosine learning-rate schedule, and organism-weighted MSE. Each observation receives weight `N / (K * count[organism])`, giving each species equal total weight within the supplied training rows.

The standalone entry point is:

```bash
python model/train_and_optimize/train.py
```

It trains on **all rows** of `dataset/Antibiofilm data.csv`, without filtering the `train` flag, and writes to `antibiofilm checkpoint/antibiofilm_model.pth`. This overwrites the bundled model and reuses any encoder/scaler already in that directory. Use the Python example above for a separate training run.

### Hyperparameter optimization and cross-validation

Use `FullModelObjective` for the FiLM model or `FeedForwardObjective` for MLP variants. For example:

```python
import pandas as pd

from model.train_and_optimize.optimizer import FullModelObjective, do_study

dataset = pd.read_csv("dataset/split_by_compound/compound_split_0.csv")
dataset = dataset[dataset["train"] == True].copy()

study = do_study(
    dataset=dataset,
    database="sqlite:///experiments/database.db",
    name="full_model_max_rmse",
    n_trials=50,
    objective=FullModelObjective,
    features_start=6,
    organism_column="target_organism",
    output_column="pIC50",
    fold_column="kfold",
    normalize_features=True,
    normalizer_start=768,
    normalizer_end=900,
    metric="max_rmse",
    balance_organisms=True,
    include_organism_features=True,
)
print(study.best_value)
print(study.best_params)
```

Cross-validation uses the supplied fold assignments and fits scalers and organism encoders on each fold's training rows. Organisms absent from a fold's training set are encoded as all zeros during that fold's validation.

The optimization metrics are:

- `r2`: maximize the mean of the per-fold R² scores.
- `max_rmse`: minimize the worst species-specific normalized RMSE across **pooled out-of-fold predictions**, computed as `max_species(RMSE_species / max(std(y_species, ddof=0), 0.1))`.

`max_rmse` requires organism labels and at least two nonmissing fold assignments. It returns one pooled score, uses equal observation weights within each species, and does not support per-fold metric callbacks. Ordinary metrics return a list of fold scores from `cross_validate`.

Optuna resumes an existing study with the same database and name. Choose a new study name when changing the metric or experiment configuration. `study.best_params` contains search-space parameters, including logarithmic dimensions, and is not directly a `train_model` keyword dictionary.

## Experiments

| Directory | Experiment |
| --- | --- |
| `01_model` | Full FiLM model |
| `02_ablation_only_molformer` | MolFormer features only |
| `03_ablation_only_moldesc` | RDKit descriptors only |
| `04_no_film` | MLP with molecular and organism inputs |
| `05_no_organism_info` | MLP without organism inputs |
| `06_model_by_species` | Separate models per species |
| `07_baseline_xgb` | XGBoost baseline |
| `08_few_shots` | Few-shot adaptation |
| `09_Murko_split` | Scaffold-split evaluation |
| `10_hugeMLP` | Large MLP baseline |

For the main model:

```bash
python experiments/01_model/01_hyperparameters_optimization.py
python experiments/01_model/02_train_and_eval.py
```

The optimization script requests 1000 trials. The evaluation script trains across ten predefined compound splits and writes `experiments/01_model/predictions.csv`; rerunning it replaces that result file. Experiment scripts contain their own fixed configurations, so inspect them before launching long runs. Analysis notebooks and saved prediction CSVs are included; Optuna databases and newly generated checkpoint/preprocessing files are generally ignored by Git.

## Tests and reproducibility

Run the focused tests for the CLI, current metrics, and cross-validation objective:

```bash
python -m pytest test/test_cli.py test/test_metrics.py test/test_max_rmse_cv.py -q
```

The complete suite can be invoked with `python -m pytest test -q`, but some older tests still import the removed `Trainer` and `cross_validate_model` APIs or use outdated function arguments. In particular, `test/test_interaction.py` still passes attention parameters such as `num_heads` that the FiLM module no longer accepts.

The repository includes ten compound splits and ten scaffold splits. Reuse these assignments and fitted preprocessing when comparing experiments. Neural-network training does not set a global random seed, so reruns can differ. Saved results are experiment artifacts; rerun the relevant scripts to evaluate changes to the implementation.

## Citation

The manuscript title is **“Species-Context Aware Quantitative Prediction of the Antibiofilm Activity of Small Molecules.”** Full citation details will be added upon publication.

## License

This project is distributed under the [MIT License](LICENSE).
