import os
import sys
from typing import Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def data_load(
              
        data_path: str,
        features_start: int | str = 0,
        features_end: Optional[int | str] = None,
        organism_encoder_path: Optional[str] = None,
        organism_column: Optional[str | int] = None,
        normalizer_path: Optional[str] = None,
        output_column: Optional[int | str] = None,
        ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    
    """
    Loads and processes data from a CSV file for downstream machine learning tasks.

    Parameters:
        data_path (str): Path to the CSV file containing the data.
        features_start (int | str, optional): Index or name of the first feature column. Defaults to 0.
        features_end (Optional[int | str], optional): Index or name of the last feature column (exclusive). If None, uses all columns to the end.
        organism_encoder_path (Optional[str], optional): Path to the file containing the organism encoder. If None, organism encoding is skipped.
        organism_column (Optional[str | int], optional): Name or index of the column containing organism information. If None, organism encoding is skipped.
        normalizer_path (Optional[str], optional): Path to the normalizer file. If provided, features are normalized. Defaults to None.
        output_column (Optional[int | str], optional): Name or index of the output/label column. If None, output labels are not extracted.

    Returns:
        tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
            - embeddings_array: Numpy array of feature embeddings.
            - organism_array: Numpy array of encoded organism information if organism_column and organism_encoder_path are provided, otherwise None.
            - output_array: Numpy array of output labels if output_column is provided, otherwise None.

    Raises:
        FileNotFoundError: If the data file does not exist.
        KeyError: If specified columns are not found in the data.
        ValueError: If there are issues with data formatting or encoding.

    Example:
        embeddings, organisms, labels = data_load(
            data_path="data.csv",
            features_start=2,
            features_end=10,
            organism_encoder_path="encoder.pkl",
            organism_column="organism",
            normalizer_path="normalizer.pkl",
            output_column="label"
        )
    """
    data = pd.read_csv(data_path)
    embeddings_array = get_features_array(data, features_start, features_end)
    if normalizer_path is not None:
        embeddings_array = normalize(embeddings_array, normalizer_path)
    organism_array = None
    if organism_column is not None:
        organism_array = encode_organism(data[organism_column], organism_encoder_path)
    output_array = None if output_column is None else data[output_column].values

    return embeddings_array, organism_array, output_array


def get_features_array(
        df: pd.DataFrame,
        features_start: int | str = 0,
        features_end: Optional[int | str] = None) -> np.ndarray:
    """
        Extracts a NumPy array of feature values from a pandas DataFrame, using either integer-based or label-based slicing.
        Parameters:
            df (pd.DataFrame): The input DataFrame containing the features.
            features_start (int | str, optional): The starting index or column label for feature selection. Defaults to 0.
            features_end (Optional[int | str], optional): The ending index or column label for feature selection. If None, selects all columns from features_start onward. Must be the same type as features_start if provided.
        Returns:
            np.ndarray: A NumPy array containing the selected feature values.
        Raises:
            ValueError: If features_end is not None and its type does not match features_start.
        Notes:
            - If features_start and features_end are integers, rows are selected by index.
            - If features_start and features_end are strings, columns are selected by label.
            - If features_end is None and features_start is a string, all columns from features_start to the end are selected.
    """
    if features_end is not None and type(features_start) != type(features_end):
        raise ValueError("features_end should be None or the same type as features_start")
    
    if isinstance(features_start, int):
        return df.loc[features_start: features_end].values    
    if features_end is None:
        return df.loc[:, features_start:].values    
    return df.loc[:, features_start:features_end]

def normalize(data: np.ndarray, normalizer_path: str) -> np.ndarray:
    """
    Normalizes the input data using a StandardScaler. If a normalizer exists at the specified path,
    it loads and applies it; otherwise, it fits a new scaler, saves it, and applies it.
    Args:
        data (np.ndarray): The data to be normalized.
        normalizer_path (str): Path to save or load the StandardScaler object.
    Returns:
        np.ndarray: The normalized data.
    """
    if os.path.exists(normalizer_path):
        normalizer: StandardScaler = joblib.load(normalizer_path)
        return normalizer.transform(data)
    
    normalizer = StandardScaler()
    output = normalizer.fit_transform(data)
    joblib.dump(normalizer, normalizer_path)
    return output

def encode_organism(data: pd.Series, encoder_path: str) -> np.ndarray:
    """
    Encodes a pandas Series of organism names using a OneHotEncoder, saving or loading the encoder as needed.

    If an encoder exists at the specified path, it is loaded and used to transform the data.
    Otherwise, a new encoder is fitted to the data, saved to the given path, and used for transformation.

    Args:
        data (pd.Series): Series containing organism names to encode.
        encoder_path (str): Path to save or load the OneHotEncoder.

    Returns:
        np.ndarray: One-hot encoded representation of the input data.

    Raises:
        ValueError: If encoding fails due to unknown categories or data issues.
    """
    if os.path.exists(encoder_path):
        encoder: OneHotEncoder = joblib.load(encoder_path)
        return encoder.transform(data.values.reshape(-1, 1)).toarray()

    encoder = OneHotEncoder(handle_unknown='error')
    output = encoder.fit_transform(data.values.reshape(-1, 1)).toarray()
    joblib.dump(encoder, encoder_path)
    return output


    

