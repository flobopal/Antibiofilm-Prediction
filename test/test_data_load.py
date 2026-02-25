import os

import pandas as pd
from script.utils import data_load

def test_only_data():
    embeddings, ocodes, labels = data_load.data_load(
        "data/test/grover.csv",
        "grover_0",
    )
    print(ocodes)
    assert embeddings.shape == (2379, 400)
    assert ocodes is None
    assert labels is None


def test_with_encoding():
    embeddings, ocodes, labels = data_load.data_load(
        "data/test/grover.csv",
        "grover_0",
        organism_encoder_path="data/test/test_encoder.pkl",
        organism_column="target_organism",
        output_column="pIC50"
    )
    print(ocodes)
    assert os.path.exists("data/test/test_encoder.pkl")
    os.remove("data/test/test_encoder.pkl")
    assert embeddings.shape == (2379, 400)
    assert ocodes.shape == (2379, 17)
    assert len(labels) == 2379

def test_encoder_persistence():
    emb1, ocodes1, label1 = embeddings, ocodes, labels = data_load.data_load(
        "data/test/grover.csv",
        "grover_0",
        organism_encoder_path="data/test/test_encoder.pkl",
        organism_column="target_organism",
    )
    emb2, ocodes2, label2 = embeddings, ocodes, labels = data_load.data_load(
        "data/test/grover.csv",
        "grover_0",
        organism_encoder_path="data/test/test_encoder.pkl",
        organism_column="target_organism",
    )
    os.remove("data/test/test_encoder.pkl")
    assert (ocodes1==ocodes2).all()

    
def test_with_normalization():
    emb1, *_ = data_load.data_load(
        "data/test/grover.csv",
        "grover_0",
    )
    emb2, *_ = data_load.data_load(
        "data/test/grover.csv",
        "grover_0",
        normalizer_path="data/test/normalizer.pkl"
    )
    assert(os.path.exists("data/test/normalizer.pkl"))
    os.remove("data/test/normalizer.pkl")
    assert emb1.shape == emb2.shape
    assert (emb1!=emb2).all()

def test_train_test_split():
    emb, ocodes, labels = data_load.data_load(
        "data/test/grover.csv",
        "grover_0",
        organism_column="target_organism",
        organism_encoder_path="data/test/test_encoder.pkl",
        normalizer_path="data/test/normalizer.pkl",
        output_column="pIC50",
        train_test_column="train",
        train_test_value="True"
    )
    df = pd.read_csv("data/test/grover.csv")
    num_train = df[df.train].shape[0]
    assert emb.shape[0] == ocodes.shape[0] == labels.shape[0] == num_train
    os.remove("data/test/test_encoder.pkl")
    os.remove("data/test/normalizer.pkl")