import pandas as pd
from typing import Optional, Tuple

def train_val_split(df: pd.DataFrame, train_perc: float, seed: int, frd_col: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_copy = df.copy()
    # Shuffle
    df_copy = df_copy.sample(frac = 1, random_state = seed)
    
    # Get size for returned training data
    train_size = int(train_perc * df_copy.shape[0])
    
    # Split data into train and test
    df_train = df_copy[0:train_size].reset_index(drop = True)
    df_val = df_copy[train_size:].reset_index(drop = True)

    if frd_col is not None:
        # Print fraud rates for both train and val
        df_train_frdrate = 100 * df_train[frd_col].sum() / df_train.shape[0]
        df_val_frdrate = 100 * df_val[frd_col].sum() / df_val.shape[0]
        print(f"The training data's fraud rate is {df_train_frdrate:.3f}% and the validation data's fraud rate is {df_val_frdrate:.3f}%")

    return df_train, df_val