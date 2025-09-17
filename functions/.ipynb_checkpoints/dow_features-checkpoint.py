from install import check_and_install
check_and_install("holidays")

import holidays
import numpy as np
import pandas as pd

def create_dow_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Adds feature related to the day of the week to a dataset

    Args:
        df: The input DataFrame
        date_col: An input datetime column in datetime64 data type

    Returns:
        A pandas DataFrame with day of the week features added
    """
    
    # Extract the day of the week to create a weekend indicator
    df["Weekend"] = np.where(df[date_col].dt.dayofweek >= 5, 1, 0) 

    # Create a holiday indicator
    us_holidays = holidays.US()
    df["Holiday"] = np.where(df[date_col].dt.date.apply(lambda x: x in us_holidays) == True, 1, 0)

    return df