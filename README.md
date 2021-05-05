# Fibroblast Growth Factor 21 as a potential biomarker for behavioural dysfunction after tackling obesity in mice
@ Created by Dr. Alisha Parveen

# Abstract
Obesity is accompanied by behavioural disorders. Exercise, dietary adjustments, or time-restricted feeding are to date the only successful long-term interventions. By restoring the balance between disturbed energy dissipation and energy intake, Fibroblast Growth Factor 21 (FGF21) is intricately connected to nutritional regulation. The purpose of the study was to assess through behavioural parameters, whether FGF21 can endure as a biomarker after tackling obesity. Therefore, after establishing a diet-induced obesity model, mice underwent an intervention strategy with either a dietary change, treadmill exercise, or time-restricted feeding. In this study, we showed that the combination of dietary change with treadmill exercise resulted in body weight reduction, improved behavioural parameters, and the lowest FGF21 concentrations. Furthermore, feature selection algorithms revealed five highly weighted features including FGF21 and body weight. In conclusion, we suggest from the different analysis procedures that FGF21 can be considered as a potential biomarker since it endures after the intervention.

# Main task
We aimed to investigate whether FGF21 can endure as a predicted biomarker in mice after counteracting obesity. This code provides an insight overview of the machine learning section.

# Workspace
MacOS Big Sur Version 11.2.3:
A CPU was used with specification as following: Intel Core i9 2,4 GHz 8-Core with memory: 64 GB 2667 MHz DDR4.

Microsoft Windows 10 Pro:
A CPU was used with specification as following: Intel(R) Core(TM) i5-4590, 3.30GHz, 3301 MHz with memory: 8 GB.

Linux Ubuntu version 18.04 LTS:
A CPU was used with specification as following: Intel® CoreTM i7-6800K, 3.40GHz x 12 with 62.8GB memory, disk 424.6GB and OS type 64-bit. 

# Platform used
MacOS Big Sur Version 11.2.3
Python 3.8.3 with Spyder 4.2.0 Python IDE

Microsoft Windows 10 Pro and Linux Ubuntu
Python 3.7 with Spyder 4.0.1 Python IDE

# Installation
Python was installed via Anaconda3
https://www.anaconda.com/products/individual


# 1st Task: Stratification from the parent data set
Input: 01_FGF21_parent_dataset.xlsx

Code: 01_data_management.py

Output: 02_Stratified_imputated_data.xlsx

# 2nd Task: Pearson's Correlation and PCA
Input: 02_Stratified_imputated_data.xlsx

Code: 02_PCA_and_Pearson.py

Output: 02_result_PCA_data.xlsx ,
        02_result_Pearson_Correlation_data.xlsx ,
        02_result_PCA_with_index.png ,
        02_result_PCA_without_index.png ,
        02_result_Pearson_Correlation.png 

# 3rd Task: Feature selection
Input: 02_Stratified_imputated_data.xlsx

Code: 03_Feature_selection.py

Output: Console

# 4th Task: Supervised machine learning
non-feature selected data set

Input: 02_Stratified_imputated_data.xlsx

Code: 04_supervised_machine_learning.py

Output: Console


feature selected data set

Input: 04_common_Feature_selection_data.xlsx

Code: 04_supervised_machine_learning.py

Output: Console
