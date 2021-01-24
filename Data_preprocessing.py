# -*- coding: utf-8 -*-
"""
Spyder Editor
@Created in Institut für Experimentelle Chirurgie, Rostock, Germany
@ Date: 12-01-2021
@author : Dr. Alisha Parveen
"""

### Load Libraries####
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

### Load machine learning Libraries###
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel


#Load the Dataset#
dataset = pd.read_excel('FGF21.xlsx')
### Check the null values###
dataset.isnull().sum()                 #############################check missing value in columns#############################
dataset.isnull().any(axis=1).sum()     ############################# check missing value in rows###############################
#dataset.drop(labels = ['Tbase no', 'WBC [*10^9/l] ', 'RBC [*10^12/l]', 'Body'], axis =1, inplace = True) ################# Drop column from the dataset ##########

#Split the dataset#
Y= dataset.iloc[:, 0:1].values
Y=pd.DataFrame(data=Y)
Y.to_excel('Y.xlsx')
X= dataset.iloc[:, 1:38].values
X=pd.DataFrame(data=X)
X.to_excel('X.xlsx')

#HANDLING MISSING DATA####
from sklearn.impute import SimpleImputer 
X = pd.read_excel('X_before_preprocessing.xlsx')
impute= SimpleImputer(missing_values = np.nan, strategy='mean')
impute.fit(X)
X = impute.transform(X)
print(X)

#FEATURE SCALING####
from sklearn.preprocessing import Normalizer
transformer = Normalizer().fit(X)
X_norm = pd.DataFrame(transformer.fit(X).transform(X), columns = X.columns)
##check##
X=X_norm
X.to_excel('X.xlsx')
