import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def calculate_column_means(df1, df2):
    """
    Calculate the column means of two DataFrames.

    Parameters:
    -----------
    df1 : pd.DataFrame
        The first DataFrame.
    df2 : pd.DataFrame
        The second DataFrame.

    Returns:
    --------
    tuple of pd.Series
        A tuple containing two Series:
        - The first Series corresponds to the column means of the first DataFrame.
        - The second Series corresponds to the column means of the second DataFrame.
    """
    mean_df1 = df1.mean(axis=0)
    mean_df2 = df2.mean(axis=0)
    return mean_df1, mean_df2

def calculate_slope(column_means):
    """
    Calculate the slope using Simple Linear Regression (SLR) on the given column means.

    Parameters:
    -----------
    column_means : pd.Series
        A Series containing the column means.

    Returns:
    --------
    float
        The slope calculated using SLR.
    """
    # Drop NaN values
    column_means = column_means.dropna()

    # Check if column_means is empty after dropping NaNs
    if column_means.empty:
        return np.nan

    x = np.arange(len(column_means)).reshape(-1, 1)
    y = column_means.values
    reg = LinearRegression().fit(x, y)
    return reg.coef_[0]

def difference_of_slopes(df1, df2):
    """
    Calculate the difference between the slopes of two DataFrames based on their column means using SLR.

    Parameters:
    -----------
    df1 : pd.DataFrame
        The first DataFrame.
    df2 : pd.DataFrame
        The second DataFrame.

    Returns:
    --------
    float
        The difference between the slopes calculated using SLR on the column means of the two DataFrames.
    """
    mean_df1, mean_df2 = calculate_column_means(df1, df2)
    slope1 = calculate_slope(mean_df1)
    slope2 = calculate_slope(mean_df2)
    return slope1 - slope2