# Main task: Waveforms characterization

### Project structure

* **FDA_Resampling/**
  * **RawData/**
  * [**Centering_by_ColumnMeans/**](https://github.com/dejaytang/Capstone_Siemens/tree/main/Coding/FDA_Resampling/Centering_by_ColumnMeans)
     *   **Python/**
         * `RawData_CardAge.py`
         * `RawData_FluidTemperature.py`
         * `RawData_FluidType.py`
     *  **HTML/**
     *  **JupyterNotebook/**
  * [**StartingPoints_ZeroAligned/**](https://github.com/dejaytang/Capstone_Siemens/tree/main/Coding/FDA_Resampling/StartingPoints_ZeroAligned)
      * **Python/**
          * `ZeroStart_CardAge.py`
          * `ZeroStart_FluidTemperature.py`
          * `ZeroStart_FluidType.py`
      * **HTML/**
      * **JupyterNotebook/**
  * `time_series_visualization.py`
  * `window_extraction.py`
  * `functionalPCA.py`
  * `functional_regression.py`

## Prerequisites

The code was written in Python and can be run on terminal using the '.py' files or Jupyter notebook using '.ipynb' files. For quick access to visualize the results 'HTML' files are included. 

### Installation

Use the  library [scikit-fda](https://fda.readthedocs.io/en/latest/): `pip install scikit-fda`

### Usage custom functions

To use the functions defined in `time_series_visualization.py`, `window_extraction.py`, `functionalPCA.py`, and `functional_regression.py` in the RawData_attribute.py and ZeroStart_attribute.py files, ensure to have them within the FDA_Resampling folder as shown in the project structure.

**General Description**

1. `time_series_visualization.py`: Include three functions created for time series visualization.
   * `plot_all_time_series`: Plot all the time series from a dataframe, where every row is a time series.
   * `plot_all_time_series_and_mean_fpca`: Plot all the time series from a data frame and an additional time series(the default is the mean of all the time series). For the additional time series, two values must be provided, 'x_new' which is the array of timestamps and 'y-new' which is the array of values frome the timeseries.
   * `plot_all_time_series_in_group`: Plot time series data from four dataframes in a 2x2 grid of subplots.

2. `window_extraction.py`: Compress four functions related with the window extraction and data preprocessing steps.
   
   * `calculate_window_values`: Calculate the start and end values for calibration and sample windows.
   * `calculate_window_data`: Extracts calibration and sample window data from a given row of time series data.
   * `Merge_data`: Merge de data from the window extraction with the attributes of interest.
   * `align_to_zero`: Aligns each column of the DataFrame to its first value (zero index) by subtracting the first column from all subsequent columns.
   * `balance_index`: Balances the dataset based on the features taking the minimum number of waveforms per bin.

4. `functionalPCA.py`:

5. `functional_regression.py`:

You can import the functions using the following code:

```bash
from time_series_visualization import plot_all_time_series, plot_all_time_series_and_mean_fpca, plot_all_time_series_in_group
from window_extraction import calculate_window_values, calculate_window_data, Merge_data, align_to_zero, balance_index
from functionalPCA import fpca_two_inputs, first_component_extraction, bootstrap, create_pc_scores_plots, visualize_regression
from functional_regression import Function_regression, coefficent_visualization
```

## Pipeline

The following diagram shows the pipeline for the main task of waveform characterization, which has four main steps: data loading, data preprocessing, windows extraction and Funntional Data Analysis.

![Pipeline](Images/Flowchart_FDA_Resampling_Final.png)

## Results

**Windows visualization**

**Functional PCA**

- Visualization of waveforms and the mean function:
- Visualization of the first two eingenfunctions (principal components):
- Visualization of the mean and boxplots of the first component:
- Visualization of the eigenvalues (scores) colored-mapping by attributes:
- Simple Linear Regression slopes comparison:

**Functional Regression**
- Visualization of Functional Regression coefficients:

