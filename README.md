![Python](https://img.shields.io/badge/python-3.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)


# Antibiofilm Prediction

This repository contains the code and scripts associated with the manuscript:

**"Species-Context Aware Quantitative Prediction of the Antibiofilm Activity of Small Molecules"**
---

## Overview

We present a multimodal deep learning framework for the quantitative prediction of antibiofilm activity (pIC50) of small molecules across multiple bacterial species.

The model integrates:

- MolFormer embeddings (SMILES-based molecular language model)
- 217 RDKit molecular descriptors
- Organism-specific one-hot encoding
- A custom multi-head cross-attention mechanism
- A feedforward neural network for regression

This architecture enables organism-aware predictions and supports cross-species transfer learning.
---
## Installation

### Clone the repository

```bash
git clone https://github.com/flobopal/Antibiofilm-Prediction.git
cd AntibiofilmTransferPred
```

### Install dependencies
It is recommended to use the provided Conda environment file:

```bash
conda env create -f environment.yml
conda activate antibiofilm
```

All models were trained using PyTorch 2.x with CUDA 11.x support.

## Generating Predictions 



### Data preparation

Input data must be provided as a .csv file containing:

- **Column 0:** Index
- **Column 1:** Target organism
- **Column 2:** Compund Smiles
- **Columns 3-765:** MolFormer embeddings (see: https://github.com/IBM/molformer#feature-extraction)
- **Columns 766-:** RDKit molecular descriptors

Once you have columns 0-765, RDKit descriptros can be calculated using

```bash
python main.py descriptors "input_csv_path", "smiles_column_name", "output_csv_path"

```

or using `script.utils.molecular_descriptors.get_descriptors` function

### Running Predictions

Model checkpoints, normalization parameters, and organism encoders are stored in the `antibiofilm_checkpoint` directory.

#### From command line interface

```bash
python main.py prediction "input_csv_path", "organism_column_name", "output_csv_path"

```

Optional arguments:
```bash
python main.py prediction "input_csv_path", "organism_column_name", "output_csv_path" \
    --features_start "index of the first features column" \
    --features_end "index of the first column that does not contains features" \
    --normalizer_start "index of the first feature that needs to be normalized" \
    --normalizer_end "index of the first feature that does not need to be normalized" \
    --organism_encode_path "Path to the encoder" \
    --normalizer_path "Path to the normalizer" \
    --model_checkpoint_path "Path to the model checkpoint"

```

Arguments `--features-start`, `--features-end`, `normalizer-start` and `normalizer-end` are only necessary if the input file does not follow the structure described in the **Data preparation** section. If `features-end` or `normalizer-end` are not indicated, the script will read all columns from the corresponding -start index to the end.

By default, the script will load the organism encoder, the normalizer and the model checkpoint located at the `antibiofilm_checkpoint` directory. If they are located elsewhere, their paths can be indicated using `--organism_encode_path`, `--normalizer_path` and `--model_checkpoint_path`.

#### Using Python code

Example script: 

`experiments/01 Model Train and evaluation/03 eval.py`

```python
from script.utils.data_load import data_load
from model.full_model import FullModel

# Data load
Xd, Xp, _ = data_load(
    "path-to-csv-file",
    features_start=xx, # column in which embeddings start

    # To normalize some of the features (optional)
    normalizer_path="path-to-save-normalizer-file", 
    normalizer_start=xx, # Column index
    normalizer_end=xx # Column index

    # To include columns with one-hot encoded organisms (optional):
    organism_encoder_path="path-to-save encoder-file", 
    organism_column='name-of-organism-column',
    )

# Model load
model = FullModel.load("path-to-model-checkpoint")

# Getting predictions
model.eval()
with torch.no_grad():
    y_pred = model.forward(Xd, Xp)

```


## General usage of the module

All scripts used in the manuscript are available in the `experiments` folder.

### Training a Model

#### Data preparation

Training data must be provided as a .csv file with the following columns:

- pIC50 (or target variable)

- Target organism

- Molecular embeddings / descriptors / fingerprints

- (Optional) Train/test split column

MolFormer embeddings can be obtained from:
https://github.com/IBM/molformer#feature-extraction

RDKit descriptors can be computed using:

`script/utils/molecular_descriptors.py`

Or using the CLI as described in the section before

#### Hyperparameters optimization
Example script:

`experiments/01 Model Train and evaluation/01 hyperparameters optimization.py`

```python
from script.utils.data_load import data_load
from model.train_and_optimize.optimizer import do_study, Objective

# Data load
Xd, Xp, y = data_load(
    "path-to-csv-file",
    features_start=xx, # column in which embeddings start
    output_column='name-of-y-column',

    # To normalize some of the features (optional)
    normalizer_path="path-to-save-normalizer-file", 
    normalizer_start=xx, # Column index
    normalizer_end=xx # Column index

    # To include columns with one-hot encoded organisms (optional):
    organism_encoder_path="path-to-save encoder-file", 
    organism_column='name-of-organism-column',

    # To include only some rows (optional)
    train_test_column='name-of-train/test-column', 
    train_test_value='desired-value-for-train/test-column'
    )

# Hyperparameters optimization
study = do_study(
    Xd, Xp, y,
    "name-of-the-study",
    100, # number of studies
    "sqlite:///path-to-sql-database",
    Objective)

print(study.best_params)
```
To reload an study:

```python
import optuna

study = optuna.load_study(
    study_name="name-of-the-study",
    storage="sqlite:///path-to-sql-database")

best_params = study.best_params
print(best_params)
```
Studies can also be inspected using `optuna-dashboard` or the VSCode Optuna extension.

#### Training

Example script:

`experiments/01 Model Train and evaluation/02 train.py`

```python
from script.utils.data_load import data_load
from model.train_and_optimize.trainer import Trainer

# Data load
Xd, Xp, y = data_load(
    "path-to-csv-file",
    features_start=xx, # column in which embeddings start
    output_column='name-of-y-column',

    # To normalize some of the features (optional)
    normalizer_path="path-to-save-normalizer-file", 
    normalizer_start=xx, # Column index
    normalizer_end=xx # Column index

    # To include columns with one-hot encoded organisms (optional):
    organism_encoder_path="path-to-save encoder-file", 
    organism_column='name-of-organism-column',

    # To include only some rows (optional)
    train_test_column='name-of-train/test-column', 
    train_test_value='desired-value-for-train/test-column'
    )

# Training
model = Trainer(
    Xd, Xp, y,
    embed_dim=2**4, # dimension of the attention Q,K and V vectors
    num_heads=2**2, # number of heads in the attention layer
    pooling="max", # pooling method for attention layer
    hidden_dims=[2**5, 2**9], # dimensions of the hidden layers in the final MLP
    activations=["tanh", "tanh", "leaky_relu"], # activation functions for hidden and final layers in MLP
    dropout=0.0311,
    lr=9.27e-5, # learning rate
    num_epochs=2**8,
    scheduler_name="cosine",
    scheduler_kwargs={"T_max": 2**8, "eta_min": 0.0001}
)

model.train("path-to-save-model")

```

## Reproducibility

- Train/test splits are explicitly defined in the dataset.

- Normalization parameters are stored and reused.

- Hyperparameter optimization results are included in the repository.

- All test-set predictions reported in the manuscript are available.

## Citation
If you use this code, please cite:
```
(Citation details will be added upon publication.)
```

## License

MIT License