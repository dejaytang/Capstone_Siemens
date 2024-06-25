# Biosensor Waveform Characterization and Window Optimization

**Team Members:** Dylan Longert, Han (Eden) Chen, Nayeli Montiel, Nan Tang, and Jinxin (Jessie) Wang.

**Capstone Partner:** Siemens Healthineers

**Background and Focus:** The epoc® Blood Analysis System, developed by Siemens Healthineers, is essential for accurate and rapid blood analysis in medical settings. This project compares two generations of this system: System 1 (current) and System 2 (new launch). The primary goal is to analyze and optimize the biosensor waveform data from both systems, focusing on window flatness evaluation, functional data analysis (FDA), and window optimization.

**Summary of Methods:**

**Waveform Characterization:**
   
- *Window Flatness Evaluation:* Assessed the flatness of the calibration and sample windows by calculating slope coefficients for various attributes (fluid temperature, fluid type, and card age). Differences in slope between System 1 and System 2 were quantified using two-tailed t-tests.
  
- *Functional Data Analysis (FDA):* Used Functional Principal Component Analysis (FPCA) to compare the main modes of variation in waveform data produced from each system by using the first principal component (FPC1) that captures the majority of the data's variability and characterizing the waveforms using the principal component scores. Additionally, Functional Regression was used to assess the influence of external factors on the waveforms, such as the fluid temperature.

**Window Optimization:**
   
Optimized the calibration and sample windows for System 2 by adjusting delimit values to minimize slope differences between System 1 and System 2. This involved looping through possible delimit values and comparing the results to the previously defined delimit values.

**Summary of Results:**

Waveform Characterization:  System 2 showed consistently higher slopes in sensor readings relative to System 1. FPCA revealed that the primary mode of variation captured over 99% of the data's variability. System 2's calibration and sample windows are generally steeper than those for System 2.
Window Optimization: The new optimized window limits for System 2 resulted in minimized slope differences, especially for fluid types Eurotrol L4, NB, and SB-3, and test cards aged 56 to 84 days.



