import pandas as pd
import numpy as np

def read_files(list_of_files: list[str]) -> list[pd.DataFrame | np.ndarray]:
    """
    Reads a list of files and returns their contents as pandas DataFrames or NumPy arrays.

    Parameters:
        list_of_files (list[str]): List of file paths to read. Supported formats are .csv and .npz.

    Returns:
        list[pd.DataFrame | np.ndarray]: A list containing the contents of each file as either a pandas DataFrame (for .csv files)
        or a NumPy ndarray (for .npz files).

    Raises:
        ValueError: If a file does not have a .csv or .npz extension.
    """
    output = []
    for filename in list_of_files:
        if filename.endswith('.csv'):
            output.append(read_csv(filename))
        elif filename.endswith('.npz'):
            ...
        else:
            raise ValueError("Only .csv and .npz files allowed")
    return output
    
def read_csv(filename: str) -> pd.DataFrame:
    """
    Reads a CSV file and returns its contents as a pandas DataFrame.

    Args:
        filename (str): The path to the CSV file to be read.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the CSV file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        pd.errors.EmptyDataError: If the file is empty.
        pd.errors.ParserError: If the file cannot be parsed as CSV.
    """
    return pd.read_csv(filename)

def read_npz(filename: str, key: str|int = 0) -> np.ndarray:
    """
    Reads a NumPy .npz file and returns the array corresponding to the specified key.

    Parameters:
        filename (str): Path to the .npz file to read.
        key (str or int, optional): Key of the array to retrieve from the .npz file. 
            If 0 (default), the first key in the file is used.

    Returns:
        np.ndarray: The NumPy array corresponding to the specified key.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        KeyError: If the specified key is not found in the .npz file.
    """
    data = np.load(filename)
    if key == 0:
        key= list(data.keys())[key]
    return data[key]

def merge(fingerprints: list[pd.DataFrame | np.ndarray]) -> pd.DataFrame:
    dataframes = []
    for fp in fingerprints:
        if isinstance(fp, pd.DataFrame):
            dataframes.append(fp)
        else:
            dataframes.append(pd.DataFrame(fp))
    return pd.concat(dataframes, axis=1)