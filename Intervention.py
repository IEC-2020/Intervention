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
dataset.head()
dataset.to_excel('dataset.xlsx')
### Check the null values###
dataset.isnull().sum() ###check missing value in columns###
dataset.isnull().any(axis=1).sum() ### check missing value in rows####
#dataset.drop(labels = ['Tbase no', 'WBC [*10^9/l] ', 'RBC [*10^12/l]', 'Body'], axis =1, inplace = True)

##### Visualize the dataset before preprocessing ######

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
transformer
X = transformer.transform(X)
X=pd.DataFrame(data=X)
X.to_excel('X.xlsx')

##### Visualization the dataset after preprocessing ######

# Generate scatter plot of independent vs Dependent variable 
#plt.style.use('ggplot') 
#fig = plt.figure(figsize = (18, 18))  
#for index, feature_name in enumerate(X.feature_names): 
#    ax = fig.add_subplot(4, 4, index + 1) 
#    ax.scatter(X.data[:, index], Y.target) 
#    ax.set_ylabel('House Price', size = 12) 
#    ax.set_xlabel(feature_name, size = 12) 
#plt.show() 
#X1.to_excel('X1.xlsx')

#PCA_without_index#
#from sklearn.decomposition import PCA
#X = pd.read_excel('X_after_preprocessing.xlsx')
Y = pd.read_excel('Y.xlsx')
pca = PCA(n_components=2, svd_solver= 'arpack')
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame (data = principalComponents, columns = ['principal component 1', 'principal component 2'])
Y_target = pd.DataFrame(data=Y)
finalDf = pd.concat([principalDf, Y_target], axis = 1)
finalDf.to_excel('finalDf_2021_beforeprocessing.xlsx')

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
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
 
#PCA_with index#
X = pd.read_excel('X_after_preprocessing.xlsx')
Y = pd.read_excel('Y.xlsx')
from sklearn.decomposition import PCA
pca = PCA(n_components=2, svd_solver= 'arpack')
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame (data = principalComponents, columns = ['principal component 1', 'principal component 2'])
Y_target = pd.DataFrame(data=Y)
finalDf = pd.concat([principalDf, Y_target], axis = 1)
finalDf.to_excel('finalDf.xlsx')

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
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


#Generation correlation between the variables in FGF21#
X_variable = pd.DataFrame(data=X)
corr = X_variable.corr(method='pearson')
fig = plt.figure(figsize=(50,45))
ax = fig.add_subplot(111)
#cax = ax.matshow(corr,annot=True, cmap='coolwarm', vmin=-1, vmax=1)
cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(X_variable.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(X_variable.columns)
ax.set_yticklabels(X_variable.columns)
plt.show() 
corr.to_excel('corr.xlsx')

#####Feature selection Algorithms ########
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

#### Load the dataset####
X = pd.read_excel('X_after_preprocessing.xlsx')
X=pd.DataFrame(data=X)
Y = pd.read_excel('Y.xlsx')

#### Ch2 square method ####
from sklearn.preprocessing import MinMaxScaler
X_norm = MinMaxScaler().fit_transform(X)
X_norm=pd.DataFrame(data=X_norm)
chi_selector = SelectKBest(chi2, k=10)
chi_selector.fit(X_norm, Y)
chi_support = chi_selector.get_support()
chi_features = X.loc[:, chi_support].columns.tolist()
print(str(len(chi_features)), 'selected features')
score_chi_selector= chi_selector.scores_

#### RFE #####
from sklearn.feature_selection import RFE
rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select = 10, step = 10, verbose = 5)
rfe_selector.fit(X_norm, Y)
rfe_support = rfe_selector.get_support()
rfe_feature = X.loc[:,rfe_support].columns.tolist()
print(str(len(rfe_feature)), 'selected features')
score_rfe_selector = rfe_selector.estimator_.coef_

##### Ridge ####
embedded_lr_selector = SelectFromModel(LogisticRegression(penalty = 'l2'), max_features = 30)
embedded_lr_selector.fit(X_norm, Y)
embedded_lr_support = embedded_lr_selector.get_support()
embedded_lr_feature = X.loc[:, embedded_lr_support].columns.tolist()
print(str(len(embedded_lr_feature)), 'selected features')
score_lr_selector = embedded_lr_selector.estimator_.coef_

##### Random Forest #####
from sklearn.ensemble import RandomForestClassifier
embedded_rf_selector = SelectFromModel(RandomForestClassifier(), max_features = 30)
embedded_rf_selector.fit(X_norm, Y)
embedded_rf_support = embedded_rf_selector.get_support()
embedded_rf_feature = X.loc[:, embedded_rf_support].columns.tolist()
print(str(len(embedded_rf_feature)), 'selected features')
score_rf_selector=embedded_rf_selector.get_support()

### put all selection together ###
feature_selection_df = pd.DataFrame ({'Feature':index, 'Chi-2':chi_support, 'RFE':rfe_support, 'LASSO':embedded_lr_support, 'Tree':embedded_rf_support })
feature_selection_df['Total'] = np.sum(feature_selection_df, axis = 1)
feature_selection_df = feature_selection_df.sort_values(['Total', 'Feature'], ascending = False)
feature_selection_df.index = range(1, len(feature_selection_df)+1)
feature_selection_df.head()
