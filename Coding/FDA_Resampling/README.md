# Main task: Waveforms characterization

### Project structure

* **FDA_Resampling/**
  * **Centering_by_ColumnMeans/**
    * `configuration.py`
    * `entity.py`
    * `performanceanalyzer.py`
  * **StartingPoints_ZeroAligned/**
    * `data_manager_factory.py`
    * `abstract_parser.py`
    * `transformer-csv_parser.py`
    * `xml_parser.py`
    * `json_parser.py`
    * `custom_exception.py`

## Prerequisites

The code was written in Python and can be run on terminal using the '.py' files or Anaconda using '.ipynb' files. For quick access to visualize the results 'HTML' files are included. 

### Installation

Use the  library [scikit-fda](https://fda.readthedocs.io/en/latest/):

  `pip install scikit-fda`

Additionally, four modules 

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
