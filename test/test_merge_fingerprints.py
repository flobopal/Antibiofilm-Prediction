import os
import pandas as pd
from script.utils import merge_fingerprints as merge_fp

def test_read_csv():
    file = "data/test/grover.csv"
    df = merge_fp.read_csv(file)
    assert isinstance(df, pd.DataFrame)

def test_read_files():
    files = os.listdir('data/test')
    files = [os.path.join('data', 'test', file) for file in files]
    data = merge_fp.read_files(files)
    assert(len(data) == len(files))

def test_merge():
    files = os.listdir('data/test')
    files = [os.path.join('data', 'test', file) for file in files]
    fingerprints = merge_fp.read_files(files)
    df = merge_fp.merge(fingerprints)
    assert df.shape[0] == fingerprints[0].shape[0]
    assert df.shape[1] == sum(fp.shape[1] for fp in fingerprints)

def test_read_folder():
    folder = os.path.join('data', 'test')
    files = [os.path.join(folder, file) for file in os.listdir(folder)]
    target = merge_fp.merge(merge_fp.read_files(files)).shape
    assert merge_fp.read_folder(folder).shape == target

