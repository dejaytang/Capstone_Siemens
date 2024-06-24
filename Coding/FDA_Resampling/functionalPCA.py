import pandas as pd
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt
from skfda.representation.grid import FDataGrid
from skfda.preprocessing.dim_reduction.projection import FPCA
import skfda
from time_series_visualization import plot_all_time_series_and_mean_fpca
from skfda.exploratory.depth import ModifiedBandDepth
from skfda.exploratory.visualization import Boxplot
import altair as alt
from time_series_visualization import plot_all_time_series_and_mean_fpca
from skfda.ml.regression import LinearRegression
import statsmodels.api as sm

def fpca_two_inputs(time_series_s1, time_series_s2, color_fpc1_s1=None, color_fpc2_s1=None, color_fpc1_s2=None, color_fpc2_s2=None):
    """
    Performs Functional Principal Component Analysis (FPCA) on two sets of time series data from different systems
    and plots the first principal component for both systems on the same graph.

    Parameters:
    time_series_s1 (pd.DataFrame): A pandas DataFrame representing time series data from System 1. Each row represents
                                   a time series and each column represents a time point.
    time_series_s2 (pd.DataFrame): A pandas DataFrame representing time series data from System 2. Each row represents
                                   a time series and each column represents a time point.
    color_fpc1_s1 (str, optional): Color for the first principal component of System 1.
    color_fpc2_s1 (str, optional): Color for the second principal component of System 1.
    color_fpc1_s2 (str, optional): Color for the first principal component of System 2.
    color_fpc2_s2 (str, optional): Color for the second principal component of System 2.

    Returns:
    tuple: A tuple containing two pandas DataFrames representing the functional principal component scores for System 1
           and System 2 respectively.

    Prints:
    - The explained variance ratio of the first principal component for both systems.
    - The functional principal component scores for each time series in both systems.
    - Plots the first principal component for both systems on the same graph.
    - Plots the first principal component of System 1 versus the first principal component of System 2.
    """
    # fpca_s1,fpca_s2,pc_scores_s1,pc_scores_s2 = fpca_component_and_score(time_series_s1, time_series_s2, color_fpc1_s1=None, color_fpc2_s1=None, color_fpc1_s2=None, color_fpc2_s2=None)
    # Convert the data matrix to an FDataGrid object
    fd_s1 = FDataGrid(data_matrix=time_series_s1, grid_points=time_series_s1.columns.astype(float)) # System 1
    fd_s2 = FDataGrid(data_matrix=time_series_s2, grid_points=time_series_s2.columns.astype(float)) # System 2

    # Apply Functional PCA for System 1
    fpca_s1 = FPCA(n_components=2, centering=True)
    fpca_s1.fit(fd_s1)
    fpc_and_scores_s1 = fpca_s1.transform(fd_s1)

    # Apply Functional PCA for System 2
    fpca_s2 = FPCA(n_components=2)
    fpca_s2.fit(fd_s2)
    fpc_and_scores_s2 = fpca_s2.transform(fd_s2)

    # --- Explain variance ratio ---

    # System 1
    print('S1 Explain variance PC1 (%): ', fpca_s1.explained_variance_ratio_[0] * 100)
    print('S1 Explain variance PC2 (%): ', fpca_s1.explained_variance_ratio_[1] * 100)
    # System 2
    print('S2 Explain variance PC1 (%): ', fpca_s2.explained_variance_ratio_[0] * 100)
    print('S2 Explain variance PC2 (%): ', fpca_s2.explained_variance_ratio_[1] * 100)

    # --- Eigenfunctions ---

    # Extract the principal components
    principal_components_s1 = fpca_s1.components_ # System 1
    principal_components_s2 = fpca_s2.components_ # System 2

    # --- Scores (eigenvalues) ---

    # System 1
    pc_scores_s1 = pd.DataFrame(fpc_and_scores_s1, columns=['PC1_Scores', 'PC2_Scores'],
                                index=[time_series_s1.index[i] for i in range(time_series_s1.shape[0])])
    # System 2
    pc_scores_s2 = pd.DataFrame(fpc_and_scores_s2, columns=['PC1_Scores', 'PC2_Scores'],
                                index=[time_series_s2.index[i] for i in range(time_series_s2.shape[0])])

    # Identify which time series (functional data object) contributes the most to each principal component

    # System 1
    max_contribution_index_pc1_s1 = np.argmax(np.abs(fpc_and_scores_s1[:, 0]))  # Index of the maximum absolute score in the first PC
    max_contribution_index_pc2_s1 = np.argmax(np.abs(fpc_and_scores_s1[:, 1]))  # Index of the maximum absolute score in the second PC

    print(f'The time series contributing most to PC1 is at index {max_contribution_index_pc1_s1} with TestID {time_series_s1.index[max_contribution_index_pc1_s1]}')
    print(f'The time series contributing most to PC2 is at index {max_contribution_index_pc2_s1} with TestID {time_series_s1.index[max_contribution_index_pc2_s1]}')

    # System 2
    max_contribution_index_pc1_s2 = np.argmax(np.abs(fpc_and_scores_s2[:, 0]))  # Index of the maximum absolute score in the first PC
    max_contribution_index_pc2_s2 = np.argmax(np.abs(fpc_and_scores_s2[:, 1]))  # Index of the maximum absolute score in the second PC

    print(f'The time series contributing most to PC1 is at index {max_contribution_index_pc1_s2} with TestID {time_series_s2.index[max_contribution_index_pc1_s2]}')
    print(f'The time series contributing most to PC2 is at index {max_contribution_index_pc2_s2} with TestID {time_series_s2.index[max_contribution_index_pc2_s2]}')

    # --- Plotting First Principal Component for both Systems ---

    # Extracting data for System 1
    x1_FPC1 = fpca_s1.components_[0].grid_points[0]
    y1_FPC1 = fpca_s1.components_[0].data_matrix[0].flatten()

    x1_FPC2 = fpca_s1.components_[1].grid_points[0]
    y1_FPC2 = fpca_s1.components_[1].data_matrix[0].flatten()

    y_s1_mean_function = fpca_s1.mean_.data_matrix.flatten()

    # Extracting data for System 2
    x2_FPC1 = fpca_s2.components_[0].grid_points[0]
    y2_FPC1 = fpca_s2.components_[0].data_matrix[0].flatten()

    x2_FPC2 = fpca_s2.components_[1].grid_points[0]
    y2_FPC2 = fpca_s2.components_[1].data_matrix[0].flatten()

    y_s2_mean_function = fpca_s2.mean_.data_matrix.flatten()

    # Calculate the global y-limits

    # Mean Function
    all_y_values_s1 = np.concatenate([time_series_s1.values.flatten(), y_s1_mean_function])
    all_y_values_s2 = np.concatenate([time_series_s2.values.flatten(), y_s2_mean_function])
    global_y_min = min(all_y_values_s1.min(), all_y_values_s2.min())
    global_y_max = max(all_y_values_s1.max(), all_y_values_s2.max())

    all_y_FPC_s1 = np.concatenate([y1_FPC1, y1_FPC2])
    all_y_FPC_s2 = np.concatenate([y2_FPC1, y2_FPC2])
    global_y_FPC_min = min(all_y_FPC_s1.min(), all_y_FPC_s2.min())
    global_y_FPC_max = max(all_y_FPC_s1.max(), all_y_FPC_s2.max())

    all_y_FPC1 = np.concatenate([y1_FPC1, y2_FPC1])
    global_y_FPC1_min = all_y_FPC1.min()
    global_y_FPC1_max = all_y_FPC1.max()

    # Round the y-limits to 2 decimal places
    global_y_min = np.around(global_y_min, 2)
    global_y_max = np.around(global_y_max, 2)
    global_y_FPC_min = np.around(global_y_FPC_min, 2)
    global_y_FPC_max = np.around(global_y_FPC_max, 2)
    global_y_FPC1_min = np.around(global_y_FPC1_min, 2)
    global_y_FPC1_max = np.around(global_y_FPC1_max, 2)

    # --- Plotting Principal Components ---

    fig, axs = plt.subplots(2, 2, figsize=(16, 6))

    # Plot for System 1 waveforms
    plot_all_time_series_and_mean_fpca(axs[0, 0], time_series_s1, 'System 1 waveforms', x1_FPC1, y_s1_mean_function)
    axs[0, 0].set_ylim(global_y_min, global_y_max)  # Set y-limits

    # Plot for System 2 waveforms
    plot_all_time_series_and_mean_fpca(axs[0, 1], time_series_s2, 'System 2 waveforms', x2_FPC1, y_s2_mean_function)
    axs[0, 1].set_ylim(global_y_min, global_y_max)  # Set y-limits

    # FPC's System 1
    axs[1, 0].plot(x1_FPC1, y1_FPC1, linestyle='-', label='FPC1', color=color_fpc1_s1)
    axs[1, 0].plot(x1_FPC2, y1_FPC2, linestyle='-', label='FPC2', color=color_fpc2_s1)
    axs[1, 0].set_ylim(global_y_FPC_min, global_y_FPC_max)  # Set y-limits
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('Functional PC Values')
    axs[1, 0].legend()
    axs[1, 0].grid(False)

    # FPC's System 2
    axs[1, 1].plot(x2_FPC1, y2_FPC1, linestyle='-', label='FPC1', color=color_fpc1_s2)
    axs[1, 1].plot(x2_FPC2, y2_FPC2, linestyle='-', label='FPC2', color=color_fpc2_s2)
    axs[1, 1].set_ylim(global_y_FPC_min, global_y_FPC_max)  # Set y-limits
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('Functional PC Values')
    axs[1, 1].legend()
    axs[1, 1].grid(False)


    # FPC1 System 1 versus FPC1 System 2
    fig = plt.figure(figsize=(7.3, 3))
    plt.plot(x1_FPC1, y1_FPC1, linestyle='-', label='FPC1 - S1')
    plt.plot(x2_FPC1, y2_FPC1, linestyle='-', label='FPC1 - S2')

    plt.legend()
    plt.grid(False)
    plt.xlabel('Time (s)')
    plt.ylabel('Functional PC1 Values')
    plt.ylim(global_y_FPC1_min, global_y_FPC1_max)
    plt.show()

    return pc_scores_s1, pc_scores_s2, fpca_s1.components_, fpca_s2.components_

def first_component_extraction(time_series_s1, time_series_s2, y_axis_min=None, y_axis_max=None):
    """
    Performs Functional Principal Component Analysis (FPCA) on two sets of time series data from different systems
    Parameters:
    time_series_s1 (pd.DataFrame): A pandas DataFrame representing time series data from System 1. Each row represents
                                   a time series and each column represents a time point.
    time_series_s2 (pd.DataFrame): A pandas DataFrame representing time series data from System 2. Each row represents
                                   a time series and each column represents a time point.
    Returns:
    tuple: A tuple containing two pandas DataFrames representing the functional first principal component for System 1
           and System 2 respectively.

    """
    # Convert the data matrix to an FDataGrid object
    fd_s1 = FDataGrid(data_matrix=time_series_s1, grid_points=time_series_s1.columns.astype(float)) # System 1
    fd_s2 = FDataGrid(data_matrix=time_series_s2, grid_points=time_series_s2.columns.astype(float)) # System 2

    # Apply Functional PCA for System 1
    fpca_s1 = FPCA(n_components=2, centering=False)
    fpca_s1.fit(fd_s1)
    fpc_and_scores_s1 = fpca_s1.transform(fd_s1)

    # Apply Functional PCA for System 2
    fpca_s2 = FPCA(n_components=2, centering=False)
    fpca_s2.fit(fd_s2)
    fpc_and_scores_s2 = fpca_s2.transform(fd_s2)

    return fpca_s1.components_[0],fpca_s2.components_[0]


def bootstrap(system1_data, system2_data, sensor, window, features="FluidTempBin", n_sim=100, random_seed=42):
    """
    Perform bootstrap resampling and Functional PCA analysis on two systems' data.

    Parameters:
    system1_data (pd.DataFrame): Data from System 1.
    system2_data (pd.DataFrame): Data from System 2.
    sensor (str): Name of the sensor.
    window (str): Name of the window.
    features (str, optional): Feature to be used for resampling and analysis. Defaults to "FluidTempBin".
    n_sim (int, optional): Number of resampling iterations. Defaults to 100.
    random_seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
    tuple: A tuple containing:
           - List of Functional PC 1 values for System 1.
           - List of Functional PC 1 values for System 2.
    """

    # Set the random seed for reproducibility
    rng = np.random.default_rng(random_seed)


    # Initialize lists to store Functional PC 1 values
    fpca_s1_list = []
    fpca_s2_list = []
    fpca_s1_list_f = []
    fpca_s2_list_f = []

    # Determine the minimum number of samples for resampling
    Resample_Value1 = system1_data[features].value_counts().min()
    Resample_Value2 = system2_data[features].value_counts().min()
    Resample_Value = min(Resample_Value1, Resample_Value2)

    # Perform bootstrap resampling
    for i in range(n_sim):
        # Randomly sample data for both systems with replacement
        Id1 = system1_data.groupby(features, group_keys=False).apply(lambda x: x.sample(Resample_Value, replace=True, random_state=rng.integers(1e9))).index
        Id2 = system2_data.groupby(features, group_keys=False).apply(lambda x: x.sample(Resample_Value, replace=True, random_state=rng.integers(1e9))).index
        Resample_data1 = system1_data.loc[Id1, :].iloc[:, :-6]
        Resample_data2 = system2_data.loc[Id2, :].iloc[:, :-6]

        # Perform Functional PCA on resampled data
        fpca_s1, fpca_s2 = first_component_extraction(Resample_data1, Resample_data2)

        fpca_s1_f = fpca_s1.data_matrix[0].flatten()
        fpca_s2_f = fpca_s2.data_matrix[0].flatten()

        # Append Functional PC 1 values to the lists
        fpca_s1_list_f.append(fpca_s1_f)
        fpca_s2_list_f.append(fpca_s2_f)

        # Store the actual FPCA results directly without adding any noise
        fpca_s1_list.append(fpca_s1.data_matrix[0])
        fpca_s2_list.append(fpca_s2.data_matrix[0])

    # Calculate upper and lower percentiles for confidence intervals directly from the empirical distribution
    upper1 = np.percentile(fpca_s1_list_f, 95, axis=0)
    lower1 = np.percentile(fpca_s1_list_f, 5, axis=0)
    x1 = np.arange(len(upper1))

    upper2 = np.percentile(fpca_s2_list_f, 95, axis=0)
    lower2 = np.percentile(fpca_s2_list_f, 5, axis=0)
    x2 = np.arange(len(upper2))

    print("Confidence Interval of 1st component")
    # Plot Functional PC 1 values and confidence intervals
    fig, axs = plt.subplots(1, 3, figsize=(18.8, 3.5))

    axs[0].plot(x1, np.mean(fpca_s1_list_f, axis=0), color='tab:blue', linewidth=1, label='Mean FPC1 S1')
    axs[0].fill_between(x1, lower1, upper1, color='tab:blue', alpha=0.2, label='95% Confidence Interval S1')

    axs[0].plot(x2, np.mean(fpca_s2_list_f, axis=0), color='tab:orange', linewidth=1, label='Mean FPC1 S2')
    axs[0].fill_between(x2, lower2, upper2, color='tab:orange', alpha=0.2, label='95% Confidence Interval S2')

    axs[0].axhline(y=0, color='black', linestyle='--', linewidth=0.5)

    # Set plot labels and title
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Functional PC 1 Values')
    axs[0].set_title(f"System 1 versus System 2 in {sensor} {window}")
    axs[0].legend()

    # Print the number of resampling iterations
    print(f"The number of sampling is {Resample_Value}")

    # Calculate global y-limits for boxplots
    all_fpca_values = np.concatenate(fpca_s1_list + fpca_s2_list)
    y_min, y_max = np.min(all_fpca_values), np.max(all_fpca_values)

    # Boxplot(to distinguish the outliers)
    print("The boxplot of 1st Component")
    box_data1 = FDataGrid(data_matrix=fpca_s1_list, grid_points=range(fpca_s1_list[0].shape[0]))
    box1 = Boxplot(box_data1, depth_method=ModifiedBandDepth(), factor=0.1, axes=axs[1])  # You can decrease the value of factor to better detect outliers
    box1.show_full_outliers = True
    box1.plot()
    axs[1].set_ylim(y_min, y_max)

    box_data2 = FDataGrid(data_matrix=fpca_s2_list, grid_points=range(fpca_s1_list[0].shape[0]))
    box2 = Boxplot(box_data2, depth_method=ModifiedBandDepth(), factor=0.1, axes=axs[2])
    box2.show_full_outliers = True
    box2.plot()
    axs[2].set_ylim(y_min, y_max)

    axs[1].set_title('Boxplot for system 1')
    axs[2].set_title('Boxplot for system 2')
    plt.tight_layout()
    plt.show()
    return fpca_s1_list_f, fpca_s2_list_f

def create_pc_scores_plots(pc_scores_s1, pc_scores_s2, features_s1, features_s2, features):
    """
    Creates scatterplots for the functional principal component scores mapping with colors the additional features for every TestID for two systems.

    Parameters:
    pc_scores_s1 (pd.DataFrame): A pandas DataFrame representing the functional principal component scores for System 1.
    pc_scores_s2 (pd.DataFrame): A pandas DataFrame representing the functional principal component scores for System 2.
    features_s1 (pd.DataFrame): A pandas DataFrame representing additional features for System 1.
    features_s2 (pd.DataFrame): A pandas DataFrame representing additional features for System 2.

    Returns:
    scatterplots: A concatenated chart containing scatterplots of the functional principal component scores combined with additional features for both systems.
    """
    # --- Scores combined with Additional Features ---
    # Reset the index, making the index a column in the DataFrame
    pc_scores_s1_reset = pc_scores_s1.reset_index()
    pc_scores_s2_reset = pc_scores_s2.reset_index()
    pc_scores_s1_reset.rename(columns={'index': 'TestID'}, inplace=True)
    pc_scores_s2_reset.rename(columns={'index': 'TestID'}, inplace=True)

    # Merging functional pc scores and additional features by TestID
    system1_scores_merged = pd.merge(pc_scores_s1_reset, features_s1, how='inner', on=['TestID'])
    system2_scores_merged = pd.merge(pc_scores_s2_reset, features_s2, how='inner', on=['TestID'])

    selected_columns = ['TestID', 'PC1_Scores','PC2_Scores', 'FluidTypeBin', 'CardAgeBin', 'FluidTempBin','FluidType']
    data_s1 = system1_scores_merged[selected_columns].copy()
    data_s2 = system2_scores_merged[selected_columns].copy()

    # --- Define a color map for fluid types ---
    color_map_temp = {
        'Below 20': '#4e79a7',
        '20-25': '#f28e2b',
        'Above 25': '#e16759',
    }

    # Combine data to get the overall min and max for both axes
    combined_data = pd.concat([data_s1, data_s2])

    # Determine the range for x and y axes
    min_x = combined_data['PC1_Scores'].min()
    max_x = combined_data['PC1_Scores'].max()
    min_y = combined_data['PC2_Scores'].min()
    max_y = combined_data['PC2_Scores'].max()

    # --- Visualizations ---
    # System 1
    if features == "FluidType":
        scatter_fluid_s1 = alt.Chart(data_s1).mark_circle().encode(
            alt.X('PC1_Scores', title="Scores FPC1", scale=alt.Scale(domain=[min_x, max_x])),
            alt.Y('PC2_Scores', title="Scores FPC2", scale=alt.Scale(domain=[min_y, max_y])),
            color=alt.Color("FluidType", scale=alt.Scale(scheme='category10'), title="Fluid"),  # Changed scheme
            tooltip=['TestID', 'PC1_Scores', 'PC2_Scores', alt.Tooltip("FluidType", title="Fluid Type")]
        ).properties(
            title='Fluid Type',
            width=580,
            height=280
        )
    else:
        scatter_fluid_s1 = alt.Chart(data_s1).mark_circle().encode(
            alt.X('PC1_Scores', title="Scores FPC1", scale=alt.Scale(domain=[min_x, max_x])),
            alt.Y('PC2_Scores', title="Scores FPC2", scale=alt.Scale(domain=[min_y, max_y])),
            color=alt.Color("FluidTypeBin", scale=alt.Scale(scheme='category10'), title="Fluid"),  # Changed scheme
            tooltip=['TestID', 'PC1_Scores', 'PC2_Scores', alt.Tooltip("FluidTypeBin", title="Fluid Type")]
        ).properties(
            title='Fluid Type',
            width=580,
            height=280
        )

    scatter_age_s1 = alt.Chart(data_s1).mark_circle().encode(
    alt.X('PC1_Scores', title="Scores FPC1", scale=alt.Scale(domain=[min_x, max_x])),
    alt.Y('PC2_Scores', title="Scores FPC2", scale=alt.Scale(domain=[min_y, max_y])),
    color=alt.Color('CardAgeBin', scale=alt.Scale(scheme='tableau10'), title="Days"),
    tooltip=['TestID', 'PC1_Scores', 'PC2_Scores', alt.Tooltip('CardAgeBin', title="Card Age")]
    ).properties(
        title='Card Age',
        width=580,
        height=280
   )


    scatter_fluidTemp_s1 = alt.Chart(data_s1).mark_circle().encode(
        alt.X('PC1_Scores', title="Scores FPC1", scale=alt.Scale(domain=[min_x, max_x])),
        alt.Y('PC2_Scores', title="Scores FPC2", scale=alt.Scale(domain=[min_y, max_y])),
        color=alt.Color('FluidTempBin', scale=alt.Scale(domain=list(color_map_temp.keys()), range=list(color_map_temp.values())), title="°C"),
        tooltip=['TestID', 'PC1_Scores', 'PC2_Scores', alt.Tooltip('FluidTempBin', title="Fluid Temperature")]
    ).properties(
        title='Fluid Temperature',
        width=580,
        height=280
    )

    # System 2
    if features == "FluidType":
        scatter_fluid_s2 = alt.Chart(data_s2).mark_circle().encode(
            alt.X('PC1_Scores', title="Scores FPC1", scale=alt.Scale(domain=[min_x, max_x])),
            alt.Y('PC2_Scores', title="Scores FPC2", scale=alt.Scale(domain=[min_y, max_y])),
            color=alt.Color("FluidType", scale=alt.Scale(scheme='category10'), title="Fluid"),  # Changed scheme
            tooltip=['TestID', 'PC1_Scores', 'PC2_Scores', alt.Tooltip("FluidType", title="Fluid Type")]
        ).properties(
            title='Fluid Type',
            width=580,
            height=280
        )
    else:
        scatter_fluid_s2 = alt.Chart(data_s2).mark_circle().encode(
            alt.X('PC1_Scores', title="Scores FPC1", scale=alt.Scale(domain=[min_x, max_x])),
            alt.Y('PC2_Scores', title="Scores FPC2", scale=alt.Scale(domain=[min_y, max_y])),
            color=alt.Color("FluidTypeBin", scale=alt.Scale(scheme='category10'), title="Fluid"),  # Changed scheme
            tooltip=['TestID', 'PC1_Scores', 'PC2_Scores', alt.Tooltip("FluidTypeBin", title="Fluid Type")]
        ).properties(
            title='Fluid Type',
            width=580,
            height=280
        )
    scatter_age_s2 = alt.Chart(data_s2).mark_circle().encode(
        alt.X('PC1_Scores', title="Scores FPC1", scale=alt.Scale(domain=[min_x, max_x])),
        alt.Y('PC2_Scores', title="Scores FPC2", scale=alt.Scale(domain=[min_y, max_y])),
        color=alt.Color('CardAgeBin', scale=alt.Scale(scheme='tableau10'), title="Days"),
        tooltip=['TestID', 'PC1_Scores', 'PC2_Scores', alt.Tooltip('CardAgeBin', title="Card Age")]
    ).properties(
        title='Card Age',
        width=580,
        height=280
    )


    scatter_fluidTemp_s2 = alt.Chart(data_s2).mark_circle().encode(
        alt.X('PC1_Scores', title="Scores FPC1", scale=alt.Scale(domain=[min_x, max_x])),
        alt.Y('PC2_Scores', title="Scores FPC2", scale=alt.Scale(domain=[min_y, max_y])),
        color=alt.Color('FluidTempBin', scale=alt.Scale(domain=list(color_map_temp.keys()), range=list(color_map_temp.values())), title="°C"),
        tooltip=['TestID', 'PC1_Scores', 'PC2_Scores', alt.Tooltip('FluidTempBin', title="Fluid Temperature")]
    ).properties(
        title='Fluid Temperature',
        width=580,
        height=280
    )


    # --- Display the plots ---
    # System 1 plots
    if features == "FluidType" or features == "FluidTypeBin":
        s1_plots = alt.hconcat(
        scatter_fluid_s1, scatter_age_s1, scatter_fluidTemp_s1
        ).resolve_scale(
            color='independent'
        ).properties(
            title='System 1'
        )

        # System 2 plots
        s2_plots = alt.hconcat(
        scatter_fluid_s2, scatter_age_s2, scatter_fluidTemp_s2
        ).resolve_scale(
            color='independent'
        ).properties(
            title='System 2'
        )
    elif features == "FluidTempBin":
        s1_plots = alt.hconcat(
        scatter_fluidTemp_s1,scatter_fluid_s1, scatter_age_s1
        ).resolve_scale(
            color='independent'
        ).properties(
            title='System 1'
        )

        # System 2 plots
        s2_plots = alt.hconcat(
        scatter_fluidTemp_s2,scatter_fluid_s2, scatter_age_s2
        ).resolve_scale(
            color='independent'
        ).properties(
            title='System 2'
        )
    else:
        s1_plots = alt.hconcat(
        scatter_age_s1,scatter_fluidTemp_s1,scatter_fluid_s1
        ).resolve_scale(
            color='independent'
        ).properties(
            title='System 1'
        )

        # System 2 plots
        s2_plots = alt.hconcat(
        scatter_age_s2,scatter_fluidTemp_s2,scatter_fluid_s2
        ).resolve_scale(
            color='independent'
        ).properties(
            title='System 2'
        )

    # Concatenating plots in same grid
    plots_vconcatenated = alt.vconcat(
        s1_plots,
        s2_plots
    ).configure_view(
        stroke=None
    )

    return plots_vconcatenated

def visualize_regression(fpca_s1, fpca_s2):
    """
    Visualize regression analysis results for two sets of functional data.

    Parameters:
    fpca_s1 (FPCA): Functional PCA analysis result for system 1.
    fpca_s2 (FPCA): Functional PCA analysis result for system 2.

    Returns:
    tuple: A tuple containing:
           - Slope of the regression line for system 1.
           - Slope of the regression line for system 2.
           - Standard error of the slope for system 1.
           - Standard error of the slope for system 2.
           - Number of data points in system 1 & 2 if they are the same.
           - p-value for the hypothesis test comparing the slopes of the two systems.
    """
    # Extract grid points and data values for both sets of data
    x1 = fpca_s1.grid_points[0]
    y1 = fpca_s1.data_matrix[0].flatten()
    x2 = fpca_s2.grid_points[0]
    y2 = fpca_s2.data_matrix[0].flatten()
    n1 = len(x1)
    n2 = len(x2)
    
    # Add a constant to the feature variables for both datasets
    x1_with_const = sm.add_constant(x1)
    x2_with_const = sm.add_constant(x2)

    # Create and fit a linear regression model for the first set of data
    model1 = sm.OLS(y1, x1_with_const).fit()

    # Create and fit a linear regression model for the second set of data
    model2 = sm.OLS(y2, x2_with_const).fit()

    # Output detailed summary of the models
    summary1 = model1.summary()
    summary2 = model2.summary()
    slope1 = model1.params[1]
    se1 = model1.bse[1]
    slope2 = model2.params[1]
    se2 = model2.bse[1]
    print(summary1)
    print(summary2)

    # Hypothesis test -- t-test
    t_stat = (slope1 - slope2) / np.sqrt(se1**2 + se2**2)
    df = (n1 + n2) - 2
    p_value = round(2 * (1 - t.cdf(np.abs(t_stat), df)), 2)
    
    # Extract predicted values from the models
    y1_pred = model1.predict(x1_with_const)
    y2_pred = model2.predict(x2_with_const)

    # Compute confidence intervals for predictions
    pred1 = model1.get_prediction(x1_with_const)
    pred_summary1 = pred1.summary_frame(alpha=0.05)  # 95% confidence interval
    ci_lower1 = pred_summary1['obs_ci_lower']
    ci_upper1 = pred_summary1['obs_ci_upper']

    pred2 = model2.get_prediction(x2_with_const)
    pred_summary2 = pred2.summary_frame(alpha=0.05)  # 95% confidence interval
    ci_lower2 = pred_summary2['obs_ci_lower']
    ci_upper2 = pred_summary2['obs_ci_upper']

    # Visualize the results
    plt.figure(figsize=(10, 6))

    # Plot results for the first set of data
    plt.scatter(x1, y1, color='blue', label='Data points 1')
    plt.plot(x1, y1_pred, color='red', linewidth=2, label='Regression line 1')
    plt.fill_between(x1, ci_lower1, ci_upper1, color='red', alpha=0.2, label='95% Confidence Interval 1')

    # Plot results for the second set of data
    plt.scatter(x2, y2, color='green', label='Data points 2')
    plt.plot(x2, y2_pred, color='orange', linewidth=2, label='Regression line 2')
    plt.fill_between(x2, ci_lower2, ci_upper2, color='orange', alpha=0.2, label='95% Confidence Interval 2')

    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()

    return slope1, slope2, se1, se2, n1, p_value

