import pandas as pd
from rdkit.Chem import Descriptors, MolFromSmiles

def get_descriptors(input_file, smiles_column, output_file, **kwargs):

    def calculate_descriptors(smiles: str):
        mol = MolFromSmiles(smiles)
        return Descriptors.CalcMolDescriptors(mol)
    df = pd.read_csv(input_file, **kwargs)
    descriptors = pd.DataFrame.from_records(df[smiles_column].apply(calculate_descriptors).values, index=df.index)

    output = pd.concat([df, descriptors], axis=1)
    output.to_csv(output_file)

get_descriptors("data/Antibiofilm Molformer.csv", "curated_smiles", "data/Antibiofilm MF+MD.csv", index_col=0)