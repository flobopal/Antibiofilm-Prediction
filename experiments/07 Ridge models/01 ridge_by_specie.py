import sys
import os
sys.path.append(os.getcwd())

from pathlib import Path
import pandas as pd
import numpy as np

from script.utils.data_load import data_load

from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

results = {}

folder = Path("experiments", "07 Ridge models")
df = pd.read_csv("experiments/Antibiofilm MF+MD.csv", index_col=0)
if not os.path.exists(folder / "organisms"):
    os.mkdir(folder / "organisms")

for specie in df.target_organism.unique():

    print(f"================ {specie.upper()} ==================")
    csv_path = folder / "organisms" / f"{specie}.csv"
    normalizer_path = folder / "organisms" / f"normalizer {specie}.pkl"
    df[df.target_organism.eq(specie)].to_csv(csv_path)

    # Train
    X, _, y = data_load(csv_path, features_start=5,
            normalizer_path=normalizer_path, output_column='pIC50',
            train_test_column='train', train_test_value='True', normalizer_start=768, normalizer_end=2000)
    
    model = RidgeCV(alphas = np.logspace(-6, 3, 50))
    model.fit(X, y)

    #Eval
    X, _, y = data_load(csv_path, features_start=5,
            normalizer_path=normalizer_path, output_column='pIC50',
            train_test_column='train', train_test_value='False', normalizer_start=768, normalizer_end=2000)
    y_pred = model.predict(X)
    results[specie] = r2_score(y, y_pred)
    print(specie, results[specie])

df = pd.DataFrame({"R²": results})

df.to_csv(folder / "results.csv")
print(df.round(3))

