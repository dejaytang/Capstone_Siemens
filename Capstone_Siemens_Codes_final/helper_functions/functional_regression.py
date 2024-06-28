import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skfda.representation.grid import FDataGrid
from skfda.representation.basis import FourierBasis
from skfda.ml.regression import LinearRegression

def Function_regression(combine_dataset, len_time=40, predictor=["Fluid_Temperature_Filled","AgeOfCardInDaysAtTimeOfTest"]):
    """
    Perform functional regression using the given data.

    Parameters:
    combine_dataset (DataFrame): The dataset containing both the time series data and predictor variables.
    len_time (int): The number of basis functions to use for the Fourier basis. Default is 40 which is the time stamp for SensorA Cal window.All possible values: [40,25,90,20]
    predictor (list): The list of predictor variables to use in the regression. Default includes 'Fluid_Temperature_Filled' and 'AgeOfCardInDaysAtTimeOfTest'.

    Returns:
    funct_reg (LinearRegression): The fitted linear regression model.
    """
    if "FluidType" in predictor:
        print("Not appliable")
        return 1
    else:
        # Extract time series data and predictor variables from the combined dataset
        windows = combine_dataset.iloc[:, :-6]
        Merge_dataset = combine_dataset.iloc[:, -6:]

        # Prepare Data
        # Transform the time series data into a list
        merged_column = [list(row) for row in windows.iloc[:, :].to_numpy()]

        # Convert the list to functional data object with Fourier basis
        Y = FDataGrid(merged_column, grid_points=range(len(merged_column[1])))
        basis = FourierBasis(n_basis=len_time)
        y = Y.coordinates[0].to_basis(basis)

        # Prepare the predictor data
        x = Merge_dataset.loc[:, predictor]

        # Fit the linear regression model
        funct_reg = LinearRegression(fit_intercept=True)
        funct_reg.fit(x, y)

        # Print model summary
        print("Model Summary:", "\n")
        intercept = funct_reg.intercept_
        print("Intercept:", intercept, "\n")
        for i in range(len(predictor)):
            coef = funct_reg.coef_[i]
            print("Coefficient of {}: {}".format(predictor[i], coef), "\n")

        return funct_reg


def coefficent_visualization(funct_reg, funct_reg2, predictor, interval, title):
    """
    Visualize the coefficients of two functional regression models over a specified interval.

    Parameters:
    funct_reg (LinearRegression): The system 1 fitted linear regression model.
    funct_reg2 (LinearRegression): The system 2 fitted linear regression model.
    predictor (list): The list of predictor variables.
    interval (slice): The slice object representing the interval of interest for the coefficients.
    title (str): The title for the overall plot.

    Returns:
    None: This function displays the plots of the coefficients.
    """
    if funct_reg == 1:
        print("Not Appliable")
    else:
        num_predictors = len(predictor)
        num_intervals = len(funct_reg.intercept_.coefficients[0][interval])

        fig, axs = plt.subplots(num_predictors + 1, 1, figsize=(8, 6*num_predictors))

        # Set the overall title for the figure
        fig.suptitle(title, fontsize=16)

        # Plot the intercepts for both systems
        axs[0].plot(np.arange(num_intervals), funct_reg.intercept_.coefficients[0][interval], label='Intercept in system 1')
        axs[0].plot(np.arange(num_intervals), funct_reg2.intercept_.coefficients[0][interval], label='Intercept in system 2')
        axs[0].legend()
        axs[0].set_title('Intercept')

        # Plot the coefficients for each predictor
        for i in range(num_predictors):
            axs[i + 1].plot(np.arange(num_intervals), funct_reg.coef_[i].coefficients[0][interval], label='Coefficient in system 1')
            axs[i + 1].plot(np.arange(num_intervals), funct_reg2.coef_[i].coefficients[0][interval], label='Coefficient in system 2')
            axs[i + 1].legend()
            axs[i + 1].set_title(predictor[i])

        # Adjust the layout to make room for the title
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
