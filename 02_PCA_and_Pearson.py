#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 13:47:51 2021

@author: alishaparveen
"""
#### Import the Libraries ######
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


#Load the Dataset#
dataset = pd.read_excel('02_Stratified_imputated_data.xlsx')

#Split the dataset#
Y= dataset.iloc[:, 0:1].values
X= dataset.iloc[:, 1:33].values


#HANDLING MISSING DATA#
#from sklearn.impute import SimpleImputer 
#impute= SimpleImputer(missing_values = np.nan, strategy='mean')
#impute.fit(X) 
#X = impute.transform(X)

#FEATURE SCALING#
from sklearn.preprocessing import Normalizer
transformer = Normalizer().fit(X)
X = transformer.transform(X)

#PCA_without_index#
from sklearn.decomposition import PCA
pca = PCA(n_components=2, svd_solver= 'arpack')
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame (data = principalComponents, columns = ['principal component 1', 'principal component 2'])
Y_target = pd.DataFrame(data=Y)
finalDf = pd.concat([principalDf, Y_target], axis = 1)
finalDf.to_excel('final_PCA.xlsx')

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1 (74.68 %)', fontsize = 15)
ax.set_ylabel('Principal Component 2 (10.40 %)', fontsize = 15)
ax.set_title('', fontsize = 20)
colors = ['r','g','b','c','m','y']
for target, color in zip(list(finalDf[0].unique()),colors):
    
    ax.scatter(
       list(finalDf[finalDf[0]==target]['principal component 1'])
               ,list(finalDf[finalDf[0]==target]['principal component 2'])
               , c = color
               , s = 50)
ax.grid()
plt.show()
print(pca.explained_variance_ratio_)

###### PCA_with_Index#####
pca = PCA(n_components=2, svd_solver= 'arpack')
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame (data = principalComponents, columns = ['principal component 1', 'principal component 2'])
Y_target = pd.DataFrame(data=Y)
finalDf = pd.concat([principalDf, Y_target], axis = 1)
finalDf.to_excel('finalDf.xlsx')

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1 (74.68 %)', fontsize = 15)
ax.set_ylabel('Principal Component 2 (10.40 %)', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
colors = ['r','g','b','c','m','y']
# lenfinalDf["principal component 1"]
count = 0
for target, color in zip(list(finalDf[0].unique()),colors):
    for x, y, index in zip(list(finalDf[finalDf[0]==target]['principal component 1']), 
                         list(finalDf[finalDf[0]==target]['principal component 2']),
                    range(len(list(finalDf[finalDf[0]==target]['principal component 2'])))
                         ):
      plt.scatter(x, y, color=color)
      plt.text(x+.01, y+.01, count, fontsize=9)
      count += 1
    # ax.scatter(
    #     list(finalDf[finalDf[0]==target]['principal component 1'])
    #            ,list(finalDf[finalDf[0]==target]['principal component 2'])
    #            , c = color
    #            , s = 50)
ax.grid()
plt.show()
print(pca.explained_variance_ratio_)


#####################################################################################################
###################################################### Perason's Correlation ########################
#####################################################################################################

#Generation correlation between the variables in FGF21#
X_variable = pd.DataFrame(data=X)  ############################# X is your preprocessed input dataset ###################
corr = X_variable.corr(method='pearson')
fig = plt.figure(figsize=(50,45))
ax = fig.add_subplot(111)
#cax = ax.matshow(corr,annot=True, cmap='coolwarm', vmin=-1, vmax=1) ################### cmap use to change the color of the heatmap #############################
cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(X_variable.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(X_variable.columns)
ax.set_yticklabels(X_variable.columns)
plt.show() 
corr.to_excel('correlation.xlsx')
