# AntibiofilmTransferPred

This repositoire contains the scripts for the paper: Specie-context aware quantitative prediction of the antibiofilm activity of small molecules 

## Overview

We present a multimodal deep learning framework for quantitative prediction of antibiofilm activity (pIC50) of small molecules across multiple bacterial species.

The model integrates:

- MolFormer embeddings (SMILES-based language model)
- 217 RDKit molecular descriptors
- Organism-specific one-hot encoding
- Custom multi-head cross-attention mechanism

The framework enables species-aware predictions and supports cross-organism transfer learning.


## Installation

- Clone the repository

```bash
git clone https://github.com/flobopal/AntibiofilmTransferPred
cd AntibiofilmTransferPred
```
- Install dependencies
```bash
git clone https://github.com/flobopal/AntibiofilmTransferPred
cd AntibiofilmTransferPred
```