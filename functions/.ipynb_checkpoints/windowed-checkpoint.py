import numpy as np
import pandas as pd
from typing import List

def windowed_count(df: pd.DataFrame, date_col: str, id_col: str, boundary: str, suffix: str, 
                   windows: List[str]) -> pd.DataFrame:
    """
    Creates a feature that counts the number of times an ID field has previously appeared in the data
    for a specified lookback window

    Args:
        df: The input pandas DataFrame
        date_col: The column of the DataFrame which contains the date with datetime type
        id_col: The column of the Dataframe which contains the ID to be searched
        boundary: Value for the "closed" parameter, which indicates whether first and/or last point in the window should                      be included in the window
        suffix: Desired suffix for the returned feature's name
        windows: The offsets (time-based durations) as a list of strings

    Returns: 
        A pandas DataFrame with the window feature for the ID appended
    """
    df_copy = df.copy()
    for duration in windows:
        dft = pd.DataFrame(df_copy[[date_col, id_col]].drop_duplicates().set_index(date_col).sort_index()\
                   .groupby(id_col)[id_col]\
                   .rolling(duration, closed = boundary)\
                   .count()\
                   .fillna(0)\
                   .astype(np.int64)\
                   .rename(id_col + "_" + duration + "_" + suffix))\
                   .reset_index()
        df_copy = pd.merge(df_copy, dft, on = [date_col, id_col], how = "inner")
    return df_copy

def windowed_fraud_count(df: pd.DataFrame, date_col: str, id_col: str, boundary: str, suffix: str, frd_col: str,
                   windows: List[str]) -> pd.DataFrame:
    """
    Creates a feature that counts the number of times an ID field has previously appeared in the data with a fraud indicator
    for a specified lookback window

    Args:
        df: The input pandas DataFrame
        date_col: The column of the DataFrame which contains the date with datetime type
        id_col: The column of the Dataframe which contains the ID to be searched
        boundary: Value for the "closed" parameter, which indicates whether first and/or last point in the window should be                       included in the window
        suffix: Desired suffix for the returned feature's name
        frd_col: The column of the DataFrame which contains the fraud indicator
        windows: The offsets (time-based durations) as a list of strings

    Returns: 
        A pandas DataFrame with the window fraud feature for the ID appended
    """
    df_copy = df.copy()
    for duration in windows:
        dft = pd.DataFrame(df_copy[[date_col, id_col, frd_col]]\
                   .drop_duplicates(subset = [date_col, id_col]).set_index(date_col).sort_index()\
                   .groupby(id_col)[frd_col]\
                   .rolling(duration, closed = boundary)\
                   .sum()\
                   .fillna(0)\
                   .astype(np.int64)\
                   .rename(id_col + "_" + duration + "_" + frd_col + suffix))\
                   .reset_index()
        df_copy = pd.merge(df_copy, dft, on = [date_col, id_col], how = "inner")
    return df_copy




    