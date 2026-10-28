import pandas as pd
import sys
import os
import json
from pathlib import Path
from collections import defaultdict
sys.path.append(os.getcwd())

from script.utils.xgb import fit_xgb_by_folds

current_path = Path(os.path.abspath(__file__)).parent

results = pd.DataFrame(columns=["split_number", "target_organism", "curated_smiles", "real_pIC50", "pred_pIC50"])
best_parameters = defaultdict(list)
if os.path.exists(current_path / "best_parameters.json"):
    with open(current_path / "best_parameters.json", "r") as file:
        best_parameters.update(json.load(file))

for partition in range(10):
    dataset = pd.read_csv(f"dataset/split_by_compound/compound_split_{partition}.csv")
    for species in dataset.target_organism.unique():
        if len(best_parameters[species]) >= partition+1:
            continue
        subdata = dataset[dataset.target_organism.eq(species)]
        subdata_train = subdata[subdata.train].copy()
        subdata_test = subdata[~subdata.train].copy()
        print("========", species.upper(), "=========")
        model = xgb_model = fit_xgb_by_folds(
            training_data=subdata_train,
            fold_column="kfold",
            features_start=6,
            output_column="pIC50",
            normalize_features=True,
            normalizer_start=768,
            normalizer_end=900,
        )
        parameters = model.best_estimator_.get_params()
        best_parameters[species].append(parameters)

        test_features = subdata_test.iloc[:,6:].to_numpy()
        subdata_test["pred_pIC50"] = model.predict(test_features)
        subdata_test["split_number"] = partition
        subdata_test.rename(columns={"pIC50": "real_pIC50"}, inplace=True)
        results = pd.concat([
            results, subdata_test[["split_number", "target_organism", "curated_smiles", "real_pIC50", "pred_pIC50"]]
        ])

        results.to_csv(current_path / "predictions.csv")
        with open(current_path / "best_parameters.json", "w") as file:
            json.dump(best_parameters, file, indent=1)