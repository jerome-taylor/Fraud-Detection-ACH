import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Tuple

def barchart_fraud_rate(df1: pd.DataFrame, df2: pd.DataFrame, df1_nm: str, df2_nm: str, cat_col: str, frd_col: str):
    """
    Creates a side-by-side bar chart to compare fraud rates of a categorical field between two datasets

    Args:
        df1: The first input data frame
        df2: The second input data frame
        df1_nm: The name to assign to the first dataframe, which will show up in the chart
        df2_nm: The name to assign to the first dataframe, which will show up in the chart
        cat_col: The column of the dataframes which contains the categorical field for comparison
        frd_col: The column of the dataframes which contains the fraud indicator
    """
    fraud_perc1 = df1.groupby(cat_col)[frd_col].mean() * 100
    fraud_perc2 = df2.groupby(cat_col)[frd_col].mean() * 100
    df_full =  pd.DataFrame({df1_nm: fraud_perc1, df2_nm: fraud_perc2})

    fig, ax = plt.subplots(figsize = (10, 6))
    width = 0.35
    x = range(len(df_full.index))

    ax.bar([i - width/2 for i in x], df_full[df1_nm], width, label = df1_nm, color = "blue")
    ax.bar([i + width/2 for i in x], df_full[df2_nm], width, label = df2_nm, color = "red") 
    
    ax.set_xlabel(cat_col)
    ax.set_ylabel("Fraud Rate")
    ax.set_title("Fraud Rate per " + cat_col)
    ax.set_xticks(x)
    ax.set_xticklabels(df_full.index, rotation = 45)
    ax.legend()
    ax.grid(axis = "y")
    plt.tight_layout()
    plt.show()

def woe_plot_encode(df1: pd.DataFrame, cat_col: str, frd_col: str, color: str, df2: Optional[pd.DataFrame] = None) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create a Weight of Evidence plot for a categorical field, then optionally encode the field to create a feature
    """
    # Get the number of frauds and nonfrauds
    num_frauds = df1[frd_col].sum()
    num_nonfrauds = df1.shape[0] - num_frauds

    # Compute WoE
    df_woe = df1.groupby(cat_col)[frd_col].agg(["count", "sum"])
    df_woe["nonfraud"] = df_woe["count"] - df_woe["sum"]
    df_woe["perc_total_fraud"] = df_woe["sum"] / num_frauds
    df_woe["perc_total_nonfraud"] = df_woe["nonfraud"] / num_nonfrauds
    df_woe["WoE"] = np.log(df_woe["perc_total_nonfraud"] / df_woe["perc_total_fraud"])
    df_woe[cat_col] = df_woe.index

    # Create the WoE Plot
    plt.figure(figsize = (14, 6))
    plt.bar(df_woe[cat_col].astype(str), df_woe["WoE"], color = color)
    plt.xlabel(cat_col)
    plt.ylabel('WoE')
    plt.title(f'Weight of Evidence (WoE) for {cat_col}')
    plt.grid(axis = "y")
    plt.xticks(rotation = 45)
    plt.tight_layout()

    if df2 is not None:
        # Encode the feature in both training and testing data
        woe_dict = df_woe["WoE"].to_dict()
        df1[cat_col + "_WoE"] = df1[cat_col].map(woe_dict)
        df2[cat_col + "_WoE"] = df2[cat_col].map(woe_dict)
        df1 = df1.drop(cat_col, axis = 1)
        df2 = df2.drop(cat_col, axis = 1)
        print(f"The feature \"{cat_col}\" has been encoded using weight of evidence")
        return df1, df2

def histogram_fraud(df1: pd.DataFrame, num_col: str, frd_col: str, num_bins: int):
    """
    Create a histogram for a numeric field based on fraud

    Args:
        df1: The input DataFrame
        num_col: The numeric column
    """
    df_frd = df1[df1[frd_col] == 1]
    df_nonfrd = df1[df1[frd_col] == 0]
    plt.hist(df_frd[num_col], bins = num_bins, color = "blue", alpha = 0.9, label = "Fraud")
    plt.hist(df_nonfrd[num_col], bins = num_bins, color = "brown", alpha = 0.3, label = "Nonfraud")
    plt.xlabel(num_col)
    plt.ylabel("Payment Count")
    plt.title("Payment Count for " + num_col)
    plt.legend()
    plt.xticks(rotation = 45)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()