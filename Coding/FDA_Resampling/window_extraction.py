import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_window_values(bubble_start, sample_start, calDelimit_input, cal_window_size_input, sampleDelimit_input, sample_window_size_input):
    """
    Calculate the start and end values for calibration and sample windows. The calculation is based on given start points and delimiters for both windows, as well as the window sizes.

    Parameters:
    -----------
    bubble_start : float
        The starting point for the bubble (reference point for calibration).
    sample_start : float
        The starting point for the sample.
    calDelimit_input : float
        The delimiter value to adjust the starting point of the calibration window.
    cal_window_size_input : float
        The size of the calibration window.
    sampleDelimit_input : float
        The delimiter value to adjust the starting point of the sample window.
    sample_window_size_input : float
        The size of the sample window.

    Returns:
    --------
    tuple of float
        A tuple containing four values:
        - cal_window_start: The starting position of the calibration window, rounded to one decimal place.
        - cal_window_end: The ending position of the calibration window, rounded to one decimal place.
        - sample_window_start: The starting position of the sample window, rounded to one decimal place.
        - sample_window_end: The ending position of the sample window, rounded to one decimal place.
    """
    cal_window_start = bubble_start - calDelimit_input
    cal_window_end = cal_window_start + cal_window_size_input
    sample_window_start = sample_start + sampleDelimit_input
    sample_window_end = sample_window_start + sample_window_size_input
    return round(cal_window_start,1), round(cal_window_end,1), round(sample_window_start,1), round(sample_window_end,1)

# Define a function to extract window data of each test
def calculate_window_data(row):
    """
    Extracts calibration and sample window data from a given row of time series data.

    Parameters:
    row (pd.Series): A pandas Series object containing time series data and window start/end times.
        The Series should have the following structure:
        - 'cal_window_start': The start time for the calibration window.
        - 'cal_window_end': The end time for the calibration window.
        - 'sample_window_start': The start time for the sample window.
        - 'sample_window_end': The end time for the sample window.
        - The index from position 22 to -4 should contain the timestamps (as float) of the time series data.

    Returns:
    tuple: A tuple containing two pandas Series:
        - The first Series corresponds to the data within the calibration window.
        - The second Series corresponds to the data within the sample window.
    """
    cal_start_time = row['cal_window_start']
    cal_end_time = row['cal_window_end']
    sample_start_time = row['sample_window_start']
    sample_end_time = row['sample_window_end']
    timestamps = row.index[22:-4].values.astype(float)
    cal_window = timestamps[(timestamps >= cal_start_time) & (timestamps <= cal_end_time)]
    sample_window = timestamps[(timestamps >= sample_start_time) & (timestamps <= sample_end_time)]
    return row[cal_window],row[sample_window]

def Merge_data(windows, merge_data):
    """
    Merge the 'windows' dataframe with selected columns from the 'merge' dataframe,
    categorize 'AgeOfCardInDaysAtTimeOfTest' and 'AmbientTemperature' into bins,
    and return a random sample of the combined dataframe.

    Parameters:
    windows (DataFrame): The primary dataframe to merge.
    merge (DataFrame): The dataframe containing the additional columns to merge.

    Returns:
    DataFrame: A random sample of 500 rows from the merged dataframe.
    """

    # Merge the two dataframes on their indices
    window_combine = windows.merge(
        merge_data[["FluidType", "AgeOfCardInDaysAtTimeOfTest", "Fluid_Temperature_Filled", "FluidTypeBin", "CardAgeBin", "FluidTempBin"]],
        how="inner",
        left_index=True,
        right_index=True
    )

    return window_combine

def align_to_zero(df):
    """
    Aligns each column of the DataFrame to its first value (zero index) by subtracting
    the first column from all subsequent columns.

    Parameters:
        df (DataFrame): The input DataFrame.

    Returns:
        DataFrame: DataFrame with each column aligned to its first value.
    """
    return df.iloc[:, 1:].sub(df.iloc[:, 1], axis=0)

def balance_index(system1_data,system2_data, features='FluidTempBin', random_state=24):
    """
    Balances the dataset based on the features, taking the total data size into account.

    Parameters:
    system1_data (pd.DataFrame): The input dataset for system 1.
    system2_data (pd.DataFrame): The input dataset for system 2.
    features (str): The column name for fluid temperature bins which you can find and update in the windows_merge.py.
    random_state (int): The seed for the random number generator to ensure reproducibility. Default is 24.

    Returns:
    tuple: A tuple containing two pandas Series objects representing the indices from the balanced datasets for system 1 and system 2 respectively.
    """
    Resample_Value1 = system1_data[features].value_counts().min()
    Resample_Value2 = system2_data[features].value_counts().min()
    Resample_Value = min(Resample_Value1, Resample_Value2)
    Id1 = system1_data.groupby(features, group_keys=False).apply(lambda x: x.sample(Resample_Value, random_state=24)).index
    Id2 = system2_data.groupby(features, group_keys=False).apply(lambda x: x.sample(Resample_Value, random_state=24)).index
    print("System1 Sensor A & B distribution:\n", system1_data.loc[Id1,features].value_counts())
    print("\n System2 Sensor A & B distribution:\n", system2_data.loc[Id2,features].value_counts())

    return Id1,Id2
