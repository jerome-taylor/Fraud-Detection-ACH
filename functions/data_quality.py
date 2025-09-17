import pandas as pd
from typing import Optional

def check_duplicates(df: pd.DataFrame, drop_ind: bool) -> Optional[pd.DataFrame]:
    """
    Checks the number of duplicate rows and optionally removes them

    Args:
        df: The dataset to be checked
        drop_ind: A boolean indicating if a new DataFrame should be returned with the duplicates dropped

    Returns:
        A Pandas DataFrame with duplicates dropped if drop_ind is True, otherwise None
    """
    num_txns = len(df)
    num_dup = num_txns - len(df.drop_duplicates())
    dup_perc = 100 * num_dup / num_txns
    print(f"Out of {num_txns:,} rows, the dataset has {num_dup:,} duplicates for a duplicate rate of:"),
    print(f"{dup_perc:.3f}%")
    if drop_ind == True:
        df = df.drop_duplicates().reset_index(drop = True)
        print(f"\nDuplicate rows have been dropped. There are {num_txns - num_dup:,} remaining.")
        return df
    else:
        return None