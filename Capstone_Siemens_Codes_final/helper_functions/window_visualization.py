import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .window_extraction import  calculate_window_values

def plot_singletest_with_windows(df, TestID, calDelimit, sampleDelimit, cal_window_size, sample_window_size):
    """
    Plots the values of a specified row in a DataFrame, marking the calibration and sample window start and end times with vertical lines.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - row_index (int): The ID of the test to plot.
    - calDelimit (decimal): The value of the new calDelimit.
    - sampleDelimit (decimal): The value of the new sampleDelimit.

    Returns:
    - Plot
    """
    row = df[df['TestID']==TestID]
    
    cal_window_start = row['cal_window_start']
    cal_window_end = row['cal_window_end']
    sample_window_start = row['sample_window_start']
    sample_window_end = row['sample_window_end']

    new_cal_window_start, new_cal_window_end, new_sample_window_start, new_sample_window_end = calculate_window_values(
        bubble_start=row['BubbleDetectTime'],
        sample_start=row['SampleDetectTime'],
        calDelimit_input=calDelimit,
        cal_window_size_input=cal_window_size,
        sampleDelimit_input=sampleDelimit,
        sample_window_size_input=sample_window_size)
    
    # Extract the time series data from the row
    timestamps = row.columns[750:-5].values.astype(float)
    values = row.iloc[:,750:-5].T.values
    
    plt.figure(figsize=(10, 6))
    
    # Plot the time series data
    plt.plot(timestamps, values, label='Time Series Data', marker='o')
    
    # Mark the calibration window
    plt.axvline(x=cal_window_start.values, color='r', linestyle='--', label='Calibration Window')
    plt.axvline(x=cal_window_end.values, color='r', linestyle='--')

    plt.axvline(x=new_cal_window_start.values, color='g', linestyle='--', label='New Calibration Window')
    plt.axvline(x=new_cal_window_end.values, color='g', linestyle='--')
    
    # Mark the sample window
    plt.axvline(x=sample_window_start.values, color='b', linestyle='--', label='Sample Window')
    plt.axvline(x=sample_window_end.values, color='b', linestyle='--')

    plt.axvline(x=new_sample_window_start.values, color='m', linestyle='--', label='New Sample Window')
    plt.axvline(x=new_sample_window_end.values, color='m', linestyle='--')

    # Adding title and labels
    plt.title(f'Test {TestID} with Calibration and Sample Windows', fontsize=16)
    plt.xlabel('Time', fontsize = 14)
    plt.ylabel('Values', fontsize = 14)
    plt.legend(prop={'size': 12})
    plt.grid(True)
    
    # Display the plot
    plt.show()

def plot_singletest_without_newWindows(df, TestID, calDelimit, sampleDelimit, cal_window_size, sample_window_size):
    """
    Plots the values of a specified row in a DataFrame, marking the calibration and sample window start and end times with vertical lines.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - row_index (int): The ID of the test to plot.
    - calDelimit (decimal): The value of the new calDelimit.
    - sampleDelimit (decimal): The value of the new sampleDelimit.

    Returns:
    - Plot
    """
    row = df[df['TestID']==TestID]
    
    cal_window_start = row['cal_window_start']
    cal_window_end = row['cal_window_end']
    sample_window_start = row['sample_window_start']
    sample_window_end = row['sample_window_end']

    new_cal_window_start, new_cal_window_end, new_sample_window_start, new_sample_window_end = calculate_window_values(
        bubble_start=row['BubbleDetectTime'],
        sample_start=row['SampleDetectTime'],
        calDelimit_input=calDelimit,
        cal_window_size_input=cal_window_size,
        sampleDelimit_input=sampleDelimit,
        sample_window_size_input=sample_window_size)
    
    # Extract the time series data from the row
    timestamps = row.columns[750:-5].values.astype(float)
    values = row.iloc[:,750:-5].T.values
    
    plt.figure(figsize=(10, 6))
    
    # Plot the time series data
    plt.plot(timestamps, values, label='Time Series Data', marker='o')
    
    # Mark the calibration window
    plt.axvline(x=cal_window_start.values, color='r', linestyle='--', label='Calibration Window')
    plt.axvline(x=cal_window_end.values, color='r', linestyle='--')

    #plt.axvline(x=new_cal_window_start.values, color='g', linestyle='--', label='New Calibration Window')
    #plt.axvline(x=new_cal_window_end.values, color='g', linestyle='--')
    
    # Mark the sample window
    plt.axvline(x=sample_window_start.values, color='b', linestyle='--', label='Sample Window')
    plt.axvline(x=sample_window_end.values, color='b', linestyle='--')

    #plt.axvline(x=new_sample_window_start.values, color='m', linestyle='--', label='New Sample Window')
    #plt.axvline(x=new_sample_window_end.values, color='m', linestyle='--')

    # Adding title and labels
    plt.title(f'Test {TestID} with Calibration and Sample Windows')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    
    # Display the plot
    plt.show()
    
