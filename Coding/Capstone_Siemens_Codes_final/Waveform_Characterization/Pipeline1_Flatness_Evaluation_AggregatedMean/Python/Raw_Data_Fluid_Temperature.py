# %%
# Change directory
import os
os.chdir("../../..")

# %%
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from helper_functions.raw_slope_functions import calculate_slopes_se, run_ttest
from helper_functions.window_extraction import process_sensor_data

# %%
# import datasets
sensorA_System1 = pd.read_csv("RawData/System1_SensorA.csv")
sensorA_System2 = pd.read_csv("RawData/System2_SensorA.csv")
sensorB_System1 = pd.read_csv("RawData/System1_SensorB.csv")
sensorB_System2 = pd.read_csv("RawData/System2_SensorB.csv")
sensorA_System1_missing = pd.read_csv("RawData/SensorA_System1_missing values.csv")
sensorA_System2_missing = pd.read_csv("RawData/SensorA_System2_missing values.csv")
keyByTestID = pd.read_csv("RawData/Key by TestID.csv")

# %%
# Transpose dataset to make columns as timestamps and rows as tests
A1_transposed = sensorA_System1.T.reset_index()
A1_transposed.columns = A1_transposed.iloc[0]
A1_transposed.rename(columns={A1_transposed.columns[0]: 'TestID'}, inplace=True)
A1_transposed = A1_transposed.drop(0)
A1_transposed['TestID'] = A1_transposed['TestID'].astype(int)

A2_transposed = sensorA_System2.T.reset_index()
A2_transposed.columns = A2_transposed.iloc[0]
A2_transposed.rename(columns={A2_transposed.columns[0]: 'TestID'}, inplace=True)
A2_transposed = A2_transposed.drop(0)
A2_transposed['TestID'] = A2_transposed['TestID'].astype(int)

A1_missing_transposed = sensorA_System1_missing.T.reset_index()
A1_missing_transposed.columns = A1_missing_transposed.iloc[0]
A1_missing_transposed.rename(columns={A1_missing_transposed.columns[0]: 'TestID'}, inplace=True)
A1_missing_transposed = A1_missing_transposed.drop(0)
A1_missing_transposed['TestID'] = A1_missing_transposed['TestID'].astype(int)

A2_missing_transposed = sensorA_System2_missing.T.reset_index()
A2_missing_transposed.columns = A2_missing_transposed.iloc[0]
A2_missing_transposed.rename(columns={A2_missing_transposed.columns[0]: 'TestID'}, inplace=True)
A2_missing_transposed = A2_missing_transposed.drop(0)
A2_missing_transposed['TestID'] = A2_missing_transposed['TestID'].astype(int)

B1_transposed = sensorB_System1.T.reset_index()
B1_transposed.columns = B1_transposed.iloc[0]
B1_transposed.rename(columns={B1_transposed.columns[0]: 'TestID'}, inplace=True)
B1_transposed = B1_transposed.drop(0)
B1_transposed['TestID'] = B1_transposed['TestID'].astype(int)

B2_transposed = sensorB_System2.T.reset_index()
B2_transposed.columns = B2_transposed.iloc[0]
B2_transposed.rename(columns={B2_transposed.columns[0]: 'TestID'}, inplace=True)
B2_transposed = B2_transposed.drop(0)
B2_transposed['TestID'] = B2_transposed['TestID'].astype(int)

# %%
# Complete A1 and A2 with the missing values
A1_transposed_mid = A1_transposed[~A1_transposed.TestID.isin(A1_missing_transposed.TestID)]
A1_transposed = pd.concat([A1_transposed_mid, A1_missing_transposed], axis=0)
A2_transposed_mid = A2_transposed[~A2_transposed.TestID.isin(A2_missing_transposed.TestID)]
A2_transposed = pd.concat([A2_transposed_mid, A2_missing_transposed], axis=0)

# %%
# Merge dataset with keyByTestID and delete unmatched tests
keyByTestID['TestID'] = keyByTestID['TestID'].astype(int)
keyByTestID['System'] = keyByTestID['System'].astype(str)
keyByTestID = keyByTestID[keyByTestID['ReturnCode'].isin(['Success','UnderReportableRange'])]

A1_keyByTestID = keyByTestID[(keyByTestID['Sensor'] == 'Sensor A') & (keyByTestID['System'] == 'System 1')]
A1_Merged = pd.merge(A1_keyByTestID,A1_transposed,how='inner', on=['TestID'])
A1_transposed = A1_transposed[A1_transposed['TestID'].isin(A1_Merged['TestID'])]

A2_keyByTestID = keyByTestID.loc[(keyByTestID['Sensor'] == 'Sensor A') & (keyByTestID['System'] != 'System 1')]
A2_Merged = pd.merge(A2_keyByTestID,A2_transposed,how='inner', on=['TestID'])
A2_transposed = A2_transposed[A2_transposed['TestID'].isin(A2_Merged['TestID'])]

sensorA_System1 = sensorA_System1.loc[:, sensorA_System1.columns.isin(A1_Merged['TestID'].astype(str))]
sensorA_System2 = sensorA_System2.loc[:, sensorA_System2.columns.isin(A2_Merged['TestID'].astype(str))]


B1_keyByTestID = keyByTestID[(keyByTestID['Sensor'] == 'Sensor B') & (keyByTestID['System'] == 'System 1')]
B1_Merged = pd.merge(B1_keyByTestID,B1_transposed,how='inner', on=['TestID'])
B1_transposed = B1_transposed[B1_transposed['TestID'].isin(B1_Merged['TestID'])]

B2_keyByTestID = keyByTestID.loc[(keyByTestID['Sensor'] == 'Sensor B') & (keyByTestID['System'] != 'System 1')]
B2_Merged = pd.merge(B2_keyByTestID,B2_transposed,how='inner', on=['TestID'])
B1_transposed = B2_transposed[B2_transposed['TestID'].isin(A2_Merged['TestID'])]

sensorB_System1 = sensorB_System1.loc[:, sensorB_System1.columns.isin(B1_Merged['TestID'].astype(str))]
sensorB_System2 = sensorB_System2.loc[:, sensorB_System2.columns.isin(B2_Merged['TestID'].astype(str))]

# %%
# Match window values of Sensor A and B for each test

# Sensor A
calDelimit = 11
cal_window_size = 8
sampleDelimit = 15
sample_window_size = 5

A1_cal_window, A1_sample_window = process_sensor_data(A1_Merged, calDelimit, cal_window_size, sampleDelimit, sample_window_size)
A2_cal_window, A2_sample_window = process_sensor_data(A2_Merged, calDelimit, cal_window_size, sampleDelimit, sample_window_size)


# sensor B
calDelimit = 20
cal_window_size = 18
sampleDelimit_blood = 24
sampleDelimit_aqueous = 30
sample_window_size = 4

B1_cal_window, B1_sample_window = process_sensor_data(B1_Merged, calDelimit, cal_window_size, sampleDelimit_blood, sample_window_size, sampleDelimit_aqueous)
B2_cal_window, B2_sample_window = process_sensor_data(B2_Merged, calDelimit, cal_window_size, sampleDelimit_blood, sample_window_size, sampleDelimit_aqueous)


# %%
# Merge attributes with extracted cal and sample window

# Sensor A
A1_attributes = A1_Merged[["TestID", "AmbientTemperature", "Fluid Temperature", "FluidType", "AgeOfCardInDaysAtTimeOfTest"]]
A1_cal_window = A1_cal_window.reset_index()
A1_cal_window_binned = A1_attributes.merge(A1_cal_window, how = "inner", on = "TestID")
A1_sample_window = A1_sample_window.reset_index()
A1_sample_window_binned = A1_attributes.merge(A1_sample_window, how = "inner", on = "TestID")

A2_attributes = A2_Merged[["TestID", "AmbientTemperature", "Fluid Temperature", "FluidType", "AgeOfCardInDaysAtTimeOfTest"]]
A2_cal_window = A2_cal_window.reset_index()
A2_cal_window_binned = A2_attributes.merge(A2_cal_window, how = "inner", on = "TestID")
A2_sample_window = A2_sample_window.reset_index()
A2_sample_window_binned = A2_attributes.merge(A2_sample_window, how = "inner", on = "TestID")

# Sensor B
B1_attributes = B1_Merged[["TestID", "AmbientTemperature", "Fluid Temperature", "FluidType", "AgeOfCardInDaysAtTimeOfTest"]]
B1_cal_window = B1_cal_window.reset_index()
B1_cal_window_binned = B1_attributes.merge(B1_cal_window, how = "inner", on = "TestID")
B1_sample_window = B1_sample_window.reset_index()
B1_sample_window_binned = B1_attributes.merge(B1_sample_window, how = "inner", on = "TestID")

B2_attributes = B2_Merged[["TestID", "AmbientTemperature", "Fluid Temperature", "FluidType", "AgeOfCardInDaysAtTimeOfTest"]]
B2_cal_window = B2_cal_window.reset_index()
B2_cal_window_binned = B2_attributes.merge(B2_cal_window, how = "inner", on = "TestID")
B2_sample_window = B2_sample_window.reset_index()
B2_sample_window_binned = B2_attributes.merge(B2_sample_window, how = "inner", on = "TestID")

# %%
# Calculate sample size for each system and sensors calibration and sample window
N_A1_cal = A1_cal_window.shape[1]
N_A2_cal = A2_cal_window.shape[1]
N_A1_sample = A1_sample_window.shape[1]
N_A2_sample = A2_sample_window.shape[1]

N_B1_cal = B1_cal_window.shape[1]
N_B2_cal = B2_cal_window.shape[1]
N_B1_sample = B1_sample_window.shape[1]
N_B2_sample = B2_sample_window.shape[1]

# %% [markdown]
# # **Fluid Temperature**

# %% [markdown]
# ## **Sensor A**

# %%
# If fluid temperature is NA, then copy Ambient Temperature into Fluid Temperature column

# A1 - cal window
for i in range(len(A1_cal_window_binned["Fluid Temperature"])):
    if np.isnan(A1_cal_window_binned["Fluid Temperature"].iloc[i]):
        A1_cal_window_binned.loc[i, "Fluid Temperature"] = A1_cal_window_binned["AmbientTemperature"].iloc[i]

# A1 - sample window
for i in range(len(A1_sample_window_binned["Fluid Temperature"])):
    if np.isnan(A1_sample_window_binned["Fluid Temperature"].iloc[i]):
        A1_sample_window_binned.loc[i, "Fluid Temperature"] = A1_sample_window_binned["AmbientTemperature"].iloc[i]

# A2 - cal window
for i in range(len(A2_cal_window_binned["Fluid Temperature"])):
    if np.isnan(A2_cal_window_binned["Fluid Temperature"].iloc[i]):
        A2_cal_window_binned.loc[i, "Fluid Temperature"] = A2_cal_window_binned["AmbientTemperature"].iloc[i]

# A2 - sample window
for i in range(len(A2_sample_window_binned["Fluid Temperature"])):
    if np.isnan(A2_sample_window_binned["Fluid Temperature"].iloc[i]):
        A2_sample_window_binned.loc[i, "Fluid Temperature"] = A2_sample_window_binned["AmbientTemperature"].iloc[i]

# %%
# Making bins and adding df column of the bin
bins = [float('-inf'), 20, 26, float('inf')]
labels = ['Under 20C', '20-26C', 'Over 26C']

# Sensor A
A1_cal_window_binned['A1-cal-ft'] = pd.cut(A1_cal_window_binned['Fluid Temperature'], bins=bins, labels=labels)
A1_sample_window_binned['A1-sample-ft'] = pd.cut(A1_sample_window_binned['Fluid Temperature'], bins=bins, labels=labels)

A2_cal_window_binned['A2-cal-ft'] = pd.cut(A2_cal_window_binned['Fluid Temperature'], bins=bins, labels=labels)
A2_sample_window_binned['A2-sample-ft'] = pd.cut(A2_sample_window_binned['Fluid Temperature'], bins=bins, labels=labels)

# %%
# Slopes - Sensor A
A1_cal = calculate_slopes_se(A1_cal_window_binned, "A1-cal-ft", labels)
A1_cal.columns = ['System-1-cal-slope', 'System-1-cal-se']

A1_sample = calculate_slopes_se(A1_sample_window_binned, "A1-sample-ft", labels)
A1_sample.columns = ['System-1-sample-slope', 'System-1-sample-se']

A2_cal = calculate_slopes_se(A2_cal_window_binned, "A2-cal-ft", labels)
A2_cal.columns = ['System-2-cal-slope', 'System-2-cal-se']

A2_sample = calculate_slopes_se(A2_sample_window_binned, "A2-sample-ft", labels)
A2_sample.columns = ['System-2-sample-slope', 'System-2-sample-se']

Sensor_A_slopes = pd.concat([A1_cal, A2_cal, A1_sample, A2_sample],axis = 1)
Sensor_A_slopes.index = labels
Sensor_A_slopes

# %%
# Plot of slopes - Sensor A
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].errorbar(range(len(labels)), Sensor_A_slopes["System-1-cal-slope"], yerr=Sensor_A_slopes["System-1-cal-se"], label="System 1", fmt='o', markersize=4, capsize=5)
axes[0].errorbar(range(len(labels)), Sensor_A_slopes["System-2-cal-slope"], yerr=Sensor_A_slopes["System-2-cal-se"], label="System 2", fmt='o', markersize=4, capsize=5)
axes[0].set_xticks(range(len(labels)))  
axes[0].set_xticklabels(labels=labels, rotation=45)
axes[0].legend(["System 1", "System 2"])
axes[0].set_title("Slope within cal window")
axes[0].set_ylabel("Slope")
axes[0].set_xlabel("Fluid Temperature")


axes[1].errorbar(range(len(labels)), Sensor_A_slopes["System-1-sample-slope"], yerr=Sensor_A_slopes["System-1-sample-se"], label="System 1", fmt='o', markersize=4, capsize=5)
axes[1].errorbar(range(len(labels)), Sensor_A_slopes["System-2-sample-slope"], yerr=Sensor_A_slopes["System-2-sample-se"], label="System 2", fmt='o', markersize=4, capsize=5)
axes[1].set_xticks(range(len(labels)))  
axes[1].set_xticklabels(labels=labels, rotation=45)
axes[1].legend(["System 1", "System 2"])
axes[1].set_title("Slope within sample window")
axes[1].set_ylabel("Slope")
axes[0].set_ylabel("Fluid Temperature")

plt.tight_layout()
plt.show()


# %%
# T-tests to compare slopes in each bin for sensor A
ttest_A = run_ttest(Sensor_A_slopes, N_A1_cal, N_A2_cal, N_A1_sample, N_A2_sample, labels)
ttest_A

# %% [markdown]
# ## **Sensor B**

# %%
# If fluid temperature is NA, then copy Ambient Temperature into Fluid Temperature column

# B1 - cal window
for i in range(len(B1_cal_window_binned["Fluid Temperature"])):
    if np.isnan(B1_cal_window_binned["Fluid Temperature"].iloc[i]):
        B1_cal_window_binned.loc[i, "Fluid Temperature"] = B1_cal_window_binned["AmbientTemperature"].iloc[i]

# B1 - sample window
for i in range(len(B1_sample_window_binned["Fluid Temperature"])):
    if np.isnan(B1_sample_window_binned["Fluid Temperature"].iloc[i]):
        B1_sample_window_binned.loc[i, "Fluid Temperature"] = B1_sample_window_binned["AmbientTemperature"].iloc[i]

# B2 - cal window
for i in range(len(B2_cal_window_binned["Fluid Temperature"])):
    if np.isnan(B2_cal_window_binned["Fluid Temperature"].iloc[i]):
        B2_cal_window_binned.loc[i, "Fluid Temperature"] = B2_cal_window_binned["AmbientTemperature"].iloc[i]

# B2 - sample window
for i in range(len(B2_sample_window_binned["Fluid Temperature"])):
    if np.isnan(B2_sample_window_binned["Fluid Temperature"].iloc[i]):
        B2_sample_window_binned.loc[i, "Fluid Temperature"] = B2_sample_window_binned["AmbientTemperature"].iloc[i]

# %%
# Making bins and adding df column of the bin
bins = [float('-inf'), 20, 26, float('inf')]
labels = ['Under 20C', '20-26C', 'Over 26C']

# Sensor B
B1_cal_window_binned['B1-cal-ft'] = pd.cut(B1_cal_window_binned['Fluid Temperature'], bins=bins, labels=labels)
B1_sample_window_binned['B1-sample-ft'] = pd.cut(B1_sample_window_binned['Fluid Temperature'], bins=bins, labels=labels)

B2_cal_window_binned['B2-cal-ft'] = pd.cut(B2_cal_window_binned['Fluid Temperature'], bins=bins, labels=labels)
B2_sample_window_binned['B2-sample-ft'] = pd.cut(B2_sample_window_binned['Fluid Temperature'], bins=bins, labels=labels)

# %%
# Slopes
B1_cal = calculate_slopes_se(B1_cal_window_binned, "B1-cal-ft", labels)
B1_cal.columns = ['System-1-cal-slope', 'System-1-cal-se']

B1_sample = calculate_slopes_se(B1_sample_window_binned, "B1-sample-ft", labels)
B1_sample.columns = ['System-1-sample-slope', 'System-1-sample-se']

B2_cal = calculate_slopes_se(B2_cal_window_binned, "B2-cal-ft", labels)
B2_cal.columns = ['System-2-cal-slope', 'System-2-cal-se']

B2_sample = calculate_slopes_se(B2_sample_window_binned, "B2-sample-ft", labels)
B2_sample.columns = ['System-2-sample-slope', 'System-2-sample-se']

Sensor_B_slopes = pd.concat([B1_cal, B2_cal, B1_sample, B2_sample],axis = 1)
Sensor_B_slopes.index = labels
Sensor_B_slopes

# %%
# Plot of slopes
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].errorbar(range(len(labels)), Sensor_B_slopes["System-1-cal-slope"], yerr=Sensor_B_slopes["System-1-cal-se"], label="System 1", fmt='o', markersize=4, capsize=5)
axes[0].errorbar(range(len(labels)), Sensor_B_slopes["System-2-cal-slope"], yerr=Sensor_B_slopes["System-2-cal-se"], label="System 2", fmt='o', markersize=4, capsize=5)
axes[0].set_xticks(range(len(labels)))  
axes[0].set_xticklabels(labels=labels, rotation=45)
axes[0].legend(["System 1", "System 2"])
axes[0].set_title("Slope within cal window")
axes[0].set_ylabel("Slope")
axes[0].set_xlabel("Fluid Temperature")


axes[1].errorbar(range(len(labels)), Sensor_B_slopes["System-1-sample-slope"], yerr=Sensor_B_slopes["System-1-sample-se"], label="System 1", fmt='o', markersize=4, capsize=5)
axes[1].errorbar(range(len(labels)), Sensor_B_slopes["System-2-sample-slope"], yerr=Sensor_B_slopes["System-2-sample-se"], label="System 2", fmt='o', markersize=4, capsize=5)
axes[1].set_xticks(range(len(labels)))  
axes[1].set_xticklabels(labels=labels, rotation=45)
axes[1].legend(["System 1", "System 2"])
axes[1].set_title("Slope within sample window")
axes[1].set_ylabel("Slope")
axes[1].set_xlabel("Fluid Temperature")

plt.tight_layout()
plt.show()


# %%
# T-tests to compare slopes in each bin
ttest_B = run_ttest(Sensor_B_slopes, N_B1_cal, N_B2_cal, N_B1_sample, N_B2_sample, labels)
ttest_B


