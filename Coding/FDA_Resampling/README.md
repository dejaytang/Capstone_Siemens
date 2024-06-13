# Main task: Waveforms characterization

### Project structure

* **FDA_Resampling/**
  * **RawData/**
    * Key by TestID.csv
    * System1_SensorA.csv
    * System1_SensorB.csv
    * System2_SensorA.csv
    * System2_SensorA.csv
    * SensorA_System1_missing values.csv
    * SensorA_System2_missing values.csv
  * **Centering_by_ColumnMeans/**
    * `RawData_CardAge.py`
    * `RawData_FluidTemperature.py`
    * `RawData_FluidType.py`
    * RawData_CardAge.ipynb
    * RawData_FluidTemperature.ipynb
    * RawData_FluidType.ipynb
    * RawData_CardAge.html
    * RawData_FluidTemperature.html
    * RawData_FluidType.html
  * **StartingPoints_ZeroAligned/**
    * `ZeroStart_CardAge.py`
    * `ZeroStart_FluidTemperature.py`
    * `ZeroStart_FluidType.py`
    * ZeroStart_CardAge.ipynb
    * ZeroStart_FluidTemperature.ipynb
    * ZeroStart_FluidType.ipynb
    * ZeroStart_CardAge.html
    * ZeroStart_FluidTemperature.html
    * ZeroStart_FluidType.html
  * `time_series_visualization.py`
  * `window_extraction.py`
  * `functionalPCA.py`
  * `functional_regression.py`

## Prerequisites

The code was written in Python and can be run on terminal using the '.py' files or Jupyter notebook using '.ipynb' files. For quick access to visualize the results 'HTML' files are included. 

### Installation

Use the  library [scikit-fda](https://fda.readthedocs.io/en/latest/):

  `pip install scikit-fda`

## Usage custom functions

To use the functions defined in `time_series_visualization.py`, `window_extraction.py`, `functionalPCA.py`, and `functional_regression.py` in the RawData_attribute.py and ZeroStart_attribute.py files, ensure to have them within the FDA_Resampling folder as shown in the project structure.

**General Description**

1. `time_series_visualization.py`: Includes three functions created for time series visualization.

  * `plot_all_time_series`: Plot all the time series from a dataframe, where every row is a time series.
  * `plot_all_time_series_and_mean_fpca`: Plot all the time series from a data frame and an additional time series. For the additional time series, two values must be provided, 'x_new' which is the array of timestamps and 'y-new' which is the array of values frome the timeseries.
  * `plot_all_time_series_in_group`: Plot time series data from four dataframes in a 2x2 grid of subplots.

2. `window_extraction.py`:

3. `functionalPCA.py`:

4. `functional_regression.py`:

You can import the functions using the following code:

```bash
from time_series_visualization import plot_all_time_series, plot_all_time_series_and_mean_fpca, plot_all_time_series_in_group
from window_extraction import calculate_window_values, calculate_window_data, Merge_data, align_to_zero, balance_index
from functionalPCA import fpca_two_inputs, first_component_extraction, bootstrap, create_pc_scores_plots, visualize_regression
from functional_regression import Function_regression, coefficent_visualization
```

## Pipeline

The following diagram shows the pipeline for the main task of waveform characterization, which is structure in four main steps: data loading, data preprocessing, windows extraction and Funntional Data Analysis.

![Pipeline](Images/Flowchart_FDA_Resampling.png)

### Data
## Results
### Windows viz
### FPCA (split into 3)
### Table slopes comparison
### Functional Regression plot
## Citation FDA Library
