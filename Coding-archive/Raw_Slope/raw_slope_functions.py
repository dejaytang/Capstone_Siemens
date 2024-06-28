import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import t


def calculate_slopes_se(bin_obj, system_cal, labels):
    """
    Calculate slopes and standard errors (SE) of linear regression models for aggregated data across different labels.

    Parameters:
    -----------
    bin_obj : pandas.DataFrame
        The DataFrame containing the data to be aggregated and analyzed.
    
    system_cal : str
        The column name in `bin_obj` to use for filtering and aggregation.
    
    labels : list
        List of unique values to filter `bin_obj`.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing two columns: 'Slope' (slopes of the linear regression models)
        and 'SE' (standard errors of the slopes) corresponding to each label in `labels`.
    """
    
    slopes = []
    std = []

    for i in labels:
        bin_filter = bin_obj.query("`{}` == @i".format(system_cal))
        bin_filter = bin_filter.drop(["TestID", "FluidType", "AmbientTemperature", "Fluid Temperature", "AgeOfCardInDaysAtTimeOfTest", system_cal], axis=1)
        agg_bin = bin_filter.mean()
        agg_bin = agg_bin.to_frame(name="mean").reset_index()

        agg_bin.reset_index(inplace=True)
        X = agg_bin['index']
        y = agg_bin['mean']
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        slopes.append(model.params[1])
        std.append(model.bse[1])
    
    result = pd.DataFrame({'Slope': slopes, 'SE': std})
    return result

def run_ttest(data, N_S1_cal, N_S2_cal, N_S1_sample, N_S2_sample, labels):
    """
    Perform independent two-sample t-tests between System-1 and System-2 based on slopes and standard errors (SE).

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing slopes and standard errors for System-1 and System-2.
        It should have columns: 'System-1-cal-slope', 'System-2-cal-slope', 'System-1-cal-se', 'System-2-cal-se',
        'System-1-sample-slope', 'System-2-sample-slope', 'System-1-sample-se', 'System-2-sample-se'.
    
    N_S1_cal : int
        Sample size for System 1 calibration window.
    
    N_S2_cal : int
        Sample size for System 2 calibration window.
    
    N_S1_sample : int
        Sample size for System 1 sample window.
    
    N_S2_sample : int
        Sample size for System 2 calibration window.
    
    labels : list
        List of unique values to filter `bin_obj`.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame with t-statistics and p-values for both calibration ('cal') and sample ('sample') datasets.
    """
    ttestcal = []
    pvalcal = []
    ttestsample = []
    pvalsample = []

    for i in range(len(data)):  
        t_stat_cal = (data["System-1-cal-slope"][i] - data["System-2-cal-slope"][i]) / np.sqrt(data["System-1-cal-se"][i]**2 + data["System-2-cal-se"][i]**2)
        df_cal = (N_S1_cal + N_S2_cal) - 2
        p_value_cal = 2 * (1 - t.cdf(np.abs(t_stat_cal), df_cal))

        t_stat_sample = (data["System-1-sample-slope"][i] - data["System-2-sample-slope"][i]) / np.sqrt(data["System-1-sample-se"][i]**2 + data["System-2-sample-se"][i]**2)
        df_sample = (N_S1_sample + N_S2_sample) - 2
        p_value_sample = 2 * (1 - t.cdf(np.abs(t_stat_sample), df_sample))

        ttestcal.append(t_stat_cal)
        pvalcal.append(p_value_cal)
        ttestsample.append(t_stat_sample)
        pvalsample.append(p_value_sample)

    result = pd.DataFrame({'t-statistic-cal': ttestcal,
                        'p-value-cal': pvalcal,
                        't-statistic-sample': ttestsample,
                        'p-value-sample': pvalsample,})

    result.index = labels
    return result