import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from typing import Optional
from itertools import product

def generate_organism_file(
        input_file: str,
        output_file: str,
        smiles_column: str,
        organisms: Optional[list[str]] = None,
        organism_encoder_path: Optional[str] = None):

    data = pd.read_csv(input_file)
    smiles = data[smiles_column].unique()
    if organisms is None and organism_encoder_path is None:
        raise ValueError("Either organisms or organism_encoder_path must be indicated")
    
    if organisms is None:
        organisms = get_organism_from_encoder(organism_encoder_path)

    df = pd.DataFrame(product(smiles, organisms), columns=['smiles', 'target_organism'])
    df.to_csv(output_file, index=0)



def get_organism_from_encoder(organism_encoder_path: str):
    encoder: OneHotEncoder = joblib.load(organism_encoder_path)
    return list(encoder.categories_[0])

