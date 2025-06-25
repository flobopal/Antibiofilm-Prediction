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

def test_merge():
    files = os.listdir('data')
    files = [os.path.join('data', file) for file in files]
    fingerprints = merge_fingerprints.read_files(files)
    df = merge_fingerprints.merge(fingerprints)
    assert df.shape[0] == fingerprints[0].shape[0]
    assert df.shape[1] == sum(fp.shape[1] for fp in fingerprints)