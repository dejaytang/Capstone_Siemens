import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def plot_all_time_series(df, title):
    """
    Plots all time series data from a DataFrame.

    Parameters:
    df (pd.DataFrame): A pandas DataFrame where each row represents a time series. The values in the rows are the
                       data points of the time series.
    title (str): The title for the plot.

    Returns:
    Display the chart with all the time series in one plot.
    """
    plt.figure(figsize=(6, 4))

    # Generate colors using a colormap
    num_lines = len(df)
    colors = plt.cm.Greys(np.linspace(0, 1, num_lines))

    # Plot each row with a different color
    for index, (i, row) in enumerate(df.iterrows()):
        plt.plot(row.values, label=f'Time Series {i + 1}', color=colors[index])

    plt.ylabel('Electrical signals (mV)')
    plt.xlabel('Time (s)')
    plt.title(title)
    plt.show()


def plot_all_time_series_and_mean_fpca(ax, df, title, x_new, y_new):
    """
    Plots all time series data from a DataFrame on a given axis and adds a new curve.

    Parameters:
    ax (matplotlib.axes.Axes): The axis to plot on.
    df (pd.DataFrame): A pandas DataFrame where each row represents a time series. The values in the rows are the
                       data points of the time series.
    title (str): The title for the plot.
    x_new (array-like): The x values for the mean from the fpca.
    y_new (array-like): The y values for the mean from the fpca.

    Returns:
    Displays the chart with all the time series and the new curve in one plot.
    """

    # Generate colors using a colormap
    num_lines = len(df)
    colors = plt.cm.Greys(np.linspace(0, 1, num_lines))

    # Plot each row with a different color
    for index, (i, row) in enumerate(df.iterrows()):
        ax.plot(row.values, color=colors[index])

    # Add the new curve
    ax.plot(x_new, y_new, label='Mean Function', color='red', linewidth=2)
    ax.legend()
    ax.set_ylabel('Electrical signals (mV)')
    ax.set_xlabel('Time (s)')
    ax.set_title(title)
    ax.grid(False)

def plot_all_time_series_in_group(df1, df2, df3, df4, lab, name1, name2, name3, name4):
    """
    Plot time series data from four dataframes in a 2x2 grid of subplots.

    Parameters:
    df1 (DataFrame): The first dataframe containing time series data.
    df2 (DataFrame): The second dataframe containing time series data.
    df3 (DataFrame): The third dataframe containing time series data.
    df4 (DataFrame): The fourth dataframe containing time series data.
    lab (str): The column name used for labeling the series.
    name1 (str): The title for the first subplot.
    name2 (str): The title for the second subplot.
    name3 (str): The title for the third subplot.
    name4 (str): The title for the fourth subplot.

    Returns:
    None: This function displays the plots of the time series data.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))  # Create a 2x2 grid of subplots

    # Create a color mapping for each unique label
    unique_labels = sorted(pd.concat([df1[lab], df2[lab], df3[lab], df4[lab]]).unique())
    colors = {label: plt.cm.tab10(i) for i, label in enumerate(unique_labels)}

    def plot_time_series(df, ax, subplot_title):
        """
        Plot the time series data on a given axis with a specified title.

        Parameters:
        df (DataFrame): The dataframe containing time series data.
        ax (Axes): The subplot axis to plot on.
        subplot_title (str): The title for the subplot.

        Returns:
        None
        """
        for index, row in df.iterrows():
            label = row[lab]
            ax.plot(range(len(row) - 6), row[:-6], label=label, color=colors[label])
        ax.set_ylabel('Electrical signals (mV)')
        ax.set_xlabel('Time (s)')

        # Create a unique legend with fixed order
        handles, labels = ax.get_legend_handles_labels()
        legend_handles = [handles[labels.index(label)] for label in unique_labels if label in labels]

        #ax.legend(legend_handles, unique_labels, title=f'{lab}')
        ax.legend(legend_handles, unique_labels)
        ax.set_title(subplot_title)

    # Plot time series data for each dataframe in the respective subplot
    plot_time_series(df1, axs[0, 0], f'{name1}')
    plot_time_series(df2, axs[0, 1], f'{name2}')
    plot_time_series(df3, axs[1, 0], f'{name3}')
    plot_time_series(df4, axs[1, 1], f'{name4}')

    # Set the overall title for the figure
    fig.suptitle(f'Windows series in different {lab}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title
    plt.savefig(f'{name1}.png')
    plt.show()
