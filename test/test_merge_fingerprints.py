import os
import pandas as pd
from script.utils import merge_fingerprints

def test_read_csv():
    file = "data/grover.csv"
    df = merge_fingerprints.read_csv(file)
    assert isinstance(df, pd.DataFrame)

def test_read_files():
    files = os.listdir('data')
    files = [os.path.join('data', file) for file in files]
    data = merge_fingerprints.read_files(files)
    assert(len(data) == len(files))