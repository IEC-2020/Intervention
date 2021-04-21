# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 16:54:36 2021

@author: Dr.alishaparveen

"""

#1. Load dataset and libraray#

import pandas as pd
import numpy as np

df_imputed = pd.read_excel("01_FGF21_parent_dataset.xlsx")
 

#2. select each group#
df1 = df_imputed.loc[df_imputed['Number_of_Experiment']==1]
df2 = df_imputed.loc[df_imputed['Number_of_Experiment']==2]
df3 = df_imputed.loc[df_imputed['Number_of_Experiment']==3]
df4 = df_imputed.loc[df_imputed['Number_of_Experiment']==4]
df5 = df_imputed.loc[df_imputed['Number_of_Experiment']==5]
df6 = df_imputed.loc[df_imputed['Number_of_Experiment']==6]


#3. Apply imputation longitundinally#
df1=df1.fillna(df1.mean())
df2=df2.fillna(df2.mean())
df3=df3.fillna(df3.mean())
df4=df4.fillna(df4.mean())
df5=df5.fillna(df5.mean())
df6=df6.fillna(df6.mean())


#4. Concatenate all the dataset#
frames = [df1,df2,df3,df4,df5,df6]
FGF21 = pd.concat(frames)

