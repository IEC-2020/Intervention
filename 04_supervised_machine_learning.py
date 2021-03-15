#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 20:28:18 2021
@author: Dr. alishaparveen
"""
###############################################################################################################################
############################################################### Necessary Libraries ###########################################
###############################################################################################################################
#For loading the dataset#
import pandas as pd
import numpy as np
# For Data preprocessing#
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Supervised machine learning algorithms#
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
# Hyper Parameter Tuning and Strata Cross validation#
from sklearn.model_selection import StratifiedKFold 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
# Generation of Statistical report of classification problem#
from sklearn.metrics import classification_report
from statistics import mean, stdev 

# Load the Dataset#
df_imputed = pd.read_excel("02_FGF21_Stratified_Imputation.xlsx")

###############################################################################################################################
############################################################### Features Preprocessing ########################################
###############################################################################################################################

Y= df_imputed.iloc[:, 0:1].values
Y= pd.DataFrame(data=Y)
X= df_imputed.iloc[:, 1:32].values
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,stratify=Y)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)
print('After Stratify Parameter is used ')
print(y_train[0].value_counts())
print(y_test[0].value_counts())
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

###############################################################################################################################
############################################################### 1. Logistics regression #######################################
###############################################################################################################################

LogR_classifier = LogisticRegression(C=1)
LogR_classifier.fit(X_train, y_train)
LogR_prediction = LogR_classifier.predict(X_test)
LogR_report = classification_report(y_test, LogR_prediction)
print(LogR_report)
skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=1) 
lst_accu_stratified = [] 
for train_index, test_index in skf.split(X, Y): 
    x_train_fold, x_test_fold = X_train, X_test 
    y_train_fold, y_test_fold = y_train, y_test
    LogR_classifier.fit(x_train_fold, y_train_fold) 
    lst_accu_stratified.append(LogR_classifier.score(x_test_fold, y_test_fold)) 
print('List of possible accuracy:', lst_accu_stratified) 
print('\nMaximum Accuracy That can be obtained from this model is:', 
      max(lst_accu_stratified)*100, '%') 
print('\nMinimum Accuracy:', 
      min(lst_accu_stratified)*100, '%') 
print('\nOverall Accuracy:', 
      mean(lst_accu_stratified)*100, '%') 

######################################################### Hyperparameter tuning ##############################################
penalty = ['l1', 'l2'] 
C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
#class_weight = [{1:0.5, 0:0.5}, {1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.7, 0:0.3}]
solver = ['liblinear', 'saga']
param_grid = dict(penalty=penalty,
                  C=C,
                  #class_weight=class_weight,
                  solver=solver)
LR_grid = GridSearchCV(estimator=LogR_classifier,
                    param_grid=param_grid,
                    scoring='accuracy',
                    verbose=1,
                    n_jobs=-1)
LR_grid_result = LR_grid.fit(X_train, y_train)
grid_LR_predictions = LR_grid_result.predict(X_test)
grid_LR_report = classification_report(y_test, grid_LR_predictions)
print(grid_LR_report)
print('Best Score: ', LR_grid_result.best_score_)
print('Best Params: ', LR_grid_result.best_params_)
skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=1) 
lst_accu_stratified = [] 
for train_index, test_index in skf.split(X, Y): 
    x_train_fold, x_test_fold = X_train, X_test 
    y_train_fold, y_test_fold = y_train, y_test
    LR_grid_result.fit(x_train_fold, y_train_fold) 
    lst_accu_stratified.append(LR_grid_result.score(x_test_fold, y_test_fold)) 
print('List of possible accuracy:', lst_accu_stratified) 
print('\nMaximum Accuracy That can be obtained from this model is:', 
      max(lst_accu_stratified)*100, '%') 
print('\nMinimum Accuracy:', 
      min(lst_accu_stratified)*100, '%') 
print('\nOverall Accuracy:', 
      mean(lst_accu_stratified)*100, '%')

###############################################################################################################################
############################################################### 2. SVC CLASSIFIER #############################################
###############################################################################################################################


svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, y_train)
svc_predictions = svclassifier.predict(X_test)
svc_report = classification_report(y_test, svc_predictions)
print(svc_report)
skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=1) 
lst_accu_stratified = [] 
for train_index, test_index in skf.split(X, Y): 
    x_train_fold, x_test_fold = X_train, X_test 
    y_train_fold, y_test_fold = y_train, y_test
    svclassifier.fit(x_train_fold, y_train_fold) 
    lst_accu_stratified.append(svclassifier.score(x_test_fold, y_test_fold)) 
print('List of possible accuracy:', lst_accu_stratified) 
print('\nMaximum Accuracy That can be obtained from this model is:', 
      max(lst_accu_stratified)*100, '%') 
print('\nMinimum Accuracy:', 
      min(lst_accu_stratified)*100, '%') 
print('\nOverall Accuracy:', 
      mean(lst_accu_stratified)*100, '%')


##########################################################Hyperparameter tuning#################################################
C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
#class_weight = [{1:0.5, 0:0.5}, {1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.7, 0:0.3}]
kernel = ['linear', 'poly', 'rbf', 'sigmoid']
gamma = ['scale', 'auto']
param_grid = dict(C=C,
                  kernel=kernel, gamma=gamma)
svc_grid = GridSearchCV(estimator=svclassifier,
                    param_grid=param_grid,
                    scoring='accuracy',
                    verbose=1,
                    n_jobs=-1)
svc_grid_result = svc_grid.fit(X_train, y_train)
grid_svc_predictions = svc_grid_result.predict(X_test)
grid_svc_report = classification_report(y_test, grid_svc_predictions)
print(grid_svc_report)
print('Best Score: ', svc_grid_result.best_score_)
print('Best Params: ', svc_grid_result.best_params_)
skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=1) 
lst_accu_stratified = [] 
for train_index, test_index in skf.split(X, Y): 
    x_train_fold, x_test_fold = X_train, X_test 
    y_train_fold, y_test_fold = y_train, y_test
    svc_grid_result.fit(x_train_fold, y_train_fold) 
    lst_accu_stratified.append(svc_grid_result.score(x_test_fold, y_test_fold)) 
print('List of possible accuracy:', lst_accu_stratified) 
print('\nMaximum Accuracy That can be obtained from this model is:', 
      max(lst_accu_stratified)*100, '%') 
print('\nMinimum Accuracy:', 
      min(lst_accu_stratified)*100, '%') 
print('\nOverall Accuracy:', 
      mean(lst_accu_stratified)*100, '%')


###############################################################################################################################
############################################################### 3. Decision Tree ##############################################
###############################################################################################################################
DTclassifier = DecisionTreeClassifier(class_weight=None, max_depth=4,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=42, splitter='best') 
DTclassifier.fit(X_train,y_train)
DT_predictions = DTclassifier.predict(X_test)
DT_report = classification_report(y_test, DT_predictions)
print(DT_report)
skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=1) 
lst_accu_stratified = [] 
for train_index, test_index in skf.split(X, Y): 
    x_train_fold, x_test_fold = X_train, X_test 
    y_train_fold, y_test_fold = y_train, y_test
    DTclassifier.fit(x_train_fold, y_train_fold) 
    lst_accu_stratified.append(DTclassifier.score(x_test_fold, y_test_fold)) 
print('List of possible accuracy:', lst_accu_stratified) 
print('\nMaximum Accuracy That can be obtained from this model is:', 
      max(lst_accu_stratified)*100, '%') 
print('\nMinimum Accuracy:', 
      min(lst_accu_stratified)*100, '%')
print('\nOverall Accuracy:', 
      mean(lst_accu_stratified)*100, '%') 

##########################################################Hyperparameter tuning#################################################
param_grid= {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4], 'criterion': ['gini', 'entropy']}
DT_grid = GridSearchCV(estimator=DTclassifier,
                    param_grid=param_grid,
                    scoring='accuracy',
                    verbose=1,
                    n_jobs=-1)
grid_DT_result = DT_grid.fit(X_train, y_train)
grid_DT_predictions = grid_DT_result.predict(X_test)
grid_DT_report = classification_report(y_test, grid_DT_predictions)
print(grid_DT_report)
print('Best Score: ', grid_DT_result.best_score_)
print('Best Params: ', grid_DT_result.best_params_)
skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=1) 
lst_accu_stratified = [] 
for train_index, test_index in skf.split(X, Y): 
    x_train_fold, x_test_fold = X_train, X_test 
    y_train_fold, y_test_fold = y_train, y_test
    DT_grid.fit(x_train_fold, y_train_fold) 
    lst_accu_stratified.append(DT_grid.score(x_test_fold, y_test_fold)) 
print('List of possible accuracy:', lst_accu_stratified) 
print('\nMaximum Accuracy That can be obtained from this model is:', 
      max(lst_accu_stratified)*100, '%') 
print('\nMinimum Accuracy:', 
      min(lst_accu_stratified)*100, '%') 
print('\nOverall Accuracy:', 
      mean(lst_accu_stratified)*100, '%')

###############################################################################################################################
############################################################### 4. Naive Bayes ################################################
###############################################################################################################################

NBclassifier = GaussianNB()
NBclassifier.fit(X_train,y_train)
NB_predictions = NBclassifier.predict(X_test)
NB_report = classification_report(y_test, NB_predictions)
print(NB_report)
skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=1) 
lst_accu_stratified = [] 
for train_index, test_index in skf.split(X, Y): 
    x_train_fold, x_test_fold = X_train, X_test 
    y_train_fold, y_test_fold = y_train, y_test
    NBclassifier.fit(x_train_fold, y_train_fold) 
    lst_accu_stratified.append(NBclassifier.score(x_test_fold, y_test_fold)) 
print('List of possible accuracy:', lst_accu_stratified) 
print('\nMaximum Accuracy That can be obtained from this model is:', 
      max(lst_accu_stratified)*100, '%') 
print('\nMinimum Accuracy:', 
      min(lst_accu_stratified)*100, '%') 
print('\nOverall Accuracy:', 
      mean(lst_accu_stratified)*100, '%') 


###############################################################################################################################
############################################################### 5. Random Forest ##############################################
###############################################################################################################################

RFClassifier= RandomForestClassifier(n_estimators= 20, max_depth=3, random_state=0)
RFClassifier.fit(X_train,y_train)
RFC_predictions = RFClassifier.predict(X_test)
RFC_report = classification_report(y_test, RFC_predictions)
print(RFC_report)
skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=1) 
lst_accu_stratified = [] 
for train_index, test_index in skf.split(X, Y): 
    x_train_fold, x_test_fold = X_train, X_test 
    y_train_fold, y_test_fold = y_train, y_test
    RFClassifier.fit(x_train_fold, y_train_fold) 
    lst_accu_stratified.append(RFClassifier.score(x_test_fold, y_test_fold)) 
print('List of possible accuracy:', lst_accu_stratified) 
print('\nMaximum Accuracy That can be obtained from this model is:', 
      max(lst_accu_stratified)*100, '%') 
print('\nMinimum Accuracy:', 
      min(lst_accu_stratified)*100, '%') 
print('\nOverall Accuracy:', 
      mean(lst_accu_stratified)*100, '%')

##########################################################Hyperparameter tuning#################################################
param_grid = { 
    'n_estimators': [100, 200, 500, 600],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}
RF_grid = GridSearchCV(estimator=RFClassifier,
                    param_grid=param_grid,
                    scoring='accuracy',
                    verbose=1,
                    n_jobs=-1)
grid_RF_result = RF_grid.fit(X_train, y_train)
grid_RF_predictions = grid_RF_result.predict(X_test)
grid_RF_report = classification_report(y_test, grid_DT_predictions)
print(grid_RF_report)
print('Best Score: ', grid_RF_result.best_score_)
print('Best Params: ', grid_RF_result.best_params_)
skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=1) 
lst_accu_stratified = [] 
for train_index, test_index in skf.split(X, Y): 
    x_train_fold, x_test_fold = X_train, X_test 
    y_train_fold, y_test_fold = y_train, y_test
    RF_grid.fit(x_train_fold, y_train_fold) 
    lst_accu_stratified.append(RF_grid.score(x_test_fold, y_test_fold)) 
print('List of possible accuracy:', lst_accu_stratified) 
print('\nMaximum Accuracy That can be obtained from this model is:', 
      max(lst_accu_stratified)*100, '%') 
print('\nMinimum Accuracy:', 
      min(lst_accu_stratified)*100, '%') 
print('\nOverall Accuracy:', 
      mean(lst_accu_stratified)*100, '%')

###############################################################################################################################
############################################################### 6. Gradient Boosting ##########################################
###############################################################################################################################

GBClassifier= GradientBoostingClassifier(n_estimators=20, 
                                         learning_rate=1.0, max_depth=3, 
                                         random_state=0).fit(X_train, y_train)
GBC_predictions = GBClassifier.predict(X_test)
GBC_report = classification_report(y_test, GBC_predictions)
print(GBC_report)
skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=1) 
lst_accu_stratified = [] 
for train_index, test_index in skf.split(X, Y): 
    x_train_fold, x_test_fold = X_train, X_test 
    y_train_fold, y_test_fold = y_train, y_test
    GBClassifier.fit(x_train_fold, y_train_fold) 
    lst_accu_stratified.append(GBClassifier.score(x_test_fold, y_test_fold)) 
print('List of possible accuracy:', lst_accu_stratified) 
print('\nMaximum Accuracy That can be obtained from this model is:', 
      max(lst_accu_stratified)*100, '%') 
print('\nMinimum Accuracy:', 
      min(lst_accu_stratified)*100, '%') 
print('\nOverall Accuracy:', 
      mean(lst_accu_stratified)*100, '%')
#################################################### Hyperparameter Tuning ####################################################
max_depth = [2,4,6,8,10,12,15,16]
criterion = ['friedman_mse', 'mse', 'mae']
n_estimators= [10,20,30,40,50,60]
max_features= ['auto', 'sqrt', 'log2']
learning_rate = [0.1, 0.001, 0.05, 0.005, 0.5, 0.0001]
param_grid = dict(max_depth=max_depth,
                  criterion=criterion,
                  n_estimators= n_estimators, max_features=max_features, learning_rate=learning_rate)

GB_grid = GridSearchCV(estimator=GBClassifier,
                    param_grid=param_grid,
                    scoring='accuracy',
                    verbose=1,
                    n_jobs=-1)
grid_GBC_result = GB_grid.fit(X_train, y_train)
grid_GBC_predictions = grid_GBC_result.predict(X_test)
grid_GBC_report = classification_report(y_test, grid_GBC_predictions)
print(grid_GBC_report)
print('Best Score: ', grid_GBC_result.best_score_)
print('Best Params: ', grid_GBC_result.best_params_)
skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=1) 
lst_accu_stratified = [] 
for train_index, test_index in skf.split(X, Y): 
    x_train_fold, x_test_fold = X_train, X_test 
    y_train_fold, y_test_fold = y_train, y_test
    GB_grid.fit(x_train_fold, y_train_fold) 
    lst_accu_stratified.append(GB_grid.score(x_test_fold, y_test_fold)) 
print('List of possible accuracy:', lst_accu_stratified) 
print('\nMaximum Accuracy That can be obtained from this model is:', 
      max(lst_accu_stratified)*100, '%') 
print('\nMinimum Accuracy:', 
      min(lst_accu_stratified)*100, '%') 
print('\nOverall Accuracy:', 
      mean(lst_accu_stratified)*100, '%')


###############################################################################################################################
############################################################### 7. SGD Classifier #############################################
###############################################################################################################################

SGDclassifier=SGDClassifier().fit(X_train, y_train)
SGD_predictions = SGDclassifier.predict(X_test)
SGD_report = classification_report(y_test, SGD_predictions)
print(SGD_report)
skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=1) 
lst_accu_stratified = [] 
for train_index, test_index in skf.split(X, Y): 
    x_train_fold, x_test_fold = X_train, X_test 
    y_train_fold, y_test_fold = y_train, y_test
    SGDclassifier.fit(x_train_fold, y_train_fold) 
    lst_accu_stratified.append(SGDclassifier.score(x_test_fold, y_test_fold)) 
print('List of possible accuracy:', lst_accu_stratified) 
print('\nMaximum Accuracy That can be obtained from this model is:', 
      max(lst_accu_stratified)*100, '%') 
print('\nMinimum Accuracy:', 
      min(lst_accu_stratified)*100, '%') 
print('\nOverall Accuracy:', 
      mean(lst_accu_stratified)*100, '%')

############################################################## Hyperparameter tuning ###########################################
param_grid = {
    'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], # learning rate
    'max_iter': [50,100,200,300,400,500, 1000], # number of epoch
    'penalty': ['l2','l1','elasticnet'],
    'n_jobs': [-1]
}
SGD_grid = GridSearchCV(estimator=SGDclassifier,
                    param_grid=param_grid,
                    scoring='accuracy',
                    verbose=1,
                    n_jobs=-1)
grid_SGD_result= SGD_grid.fit(X_train, y_train)
grid_SGD_predictions = grid_SGD_result.predict(X_test)
grid_SGD_report = classification_report(y_test, grid_SGD_predictions)
print(grid_SGD_report)
print('Best Score: ', grid_SGD_result.best_score_)
print('Best Params: ', grid_SGD_result.best_params_)
skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=1) 
lst_accu_stratified = [] 
for train_index, test_index in skf.split(X, Y): 
    x_train_fold, x_test_fold = X_train, X_test 
    y_train_fold, y_test_fold = y_train, y_test
    grid_SGD_result.fit(x_train_fold, y_train_fold) 
    lst_accu_stratified.append(grid_SGD_result.score(x_test_fold, y_test_fold)) 
print('List of possible accuracy:', lst_accu_stratified) 
print('\nMaximum Accuracy That can be obtained from this model is:', 
      max(lst_accu_stratified)*100, '%') 
print('\nMinimum Accuracy:', 
      min(lst_accu_stratified)*100, '%') 
print('\nOverall Accuracy:', 
      mean(lst_accu_stratified)*100, '%')

###############################################################################################################################
############################################################### 8. Neural Networks ############################################
###############################################################################################################################

NNclassifier = MLPClassifier().fit(X_train, y_train)
NN_predictions = NNclassifier.predict(X_test)
NN_report = classification_report(y_test, NN_predictions)
print(NN_report)
skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=1) 
lst_accu_stratified = [] 
for train_index, test_index in skf.split(X, Y): 
    x_train_fold, x_test_fold = X_train, X_test 
    y_train_fold, y_test_fold = y_train, y_test
    NNclassifier.fit(x_train_fold, y_train_fold) 
    lst_accu_stratified.append(NNclassifier.score(x_test_fold, y_test_fold)) 
print('List of possible accuracy:', lst_accu_stratified) 
print('\nMaximum Accuracy That can be obtained from this model is:', 
      max(lst_accu_stratified)*100, '%') 
print('\nMinimum Accuracy:', 
      min(lst_accu_stratified)*100, '%') 
print('\nOverall Accuracy:', 
      mean(lst_accu_stratified)*100, '%')
print('Std:', stdev(lst_accu_stratified), '%')

########################################################### Hyperparameter tuning ############################################
hidden_layer_sizes= [10,20,30,40,50,60,70,80,90]
solver = ['lbfgs', 'sgd', 'adam']
activation=['identity', 'logistic', 'tanh', 'relu']
learning_rate_init = [0.001, 0.01, 0.1, 0.005, 0.05,0.5]
max_iter= [50,100,200,300]

param_grid = dict(hidden_layer_sizes=hidden_layer_sizes,
                  solver=solver,
                  activation= activation, learning_rate_init=learning_rate_init, max_iter=max_iter)

NN_grid = GridSearchCV(estimator=NNclassifier,
                    param_grid=param_grid,
                    scoring='accuracy',
                    verbose=1,
                    n_jobs=-1)
grid_NN_result = NN_grid.fit(X_train, y_train)
grid_NN_predictions = grid_NN_result.predict(X_test)
grid_NN_report = classification_report(y_test, grid_NN_predictions)
print(grid_NN_report)
print('Best Score: ', grid_NN_result.best_score_)
print('Best Params: ', grid_NN_result.best_params_)
skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=1) 
lst_accu_stratified = [] 
for train_index, test_index in skf.split(X, Y): 
    x_train_fold, x_test_fold = X_train, X_test 
    y_train_fold, y_test_fold = y_train, y_test
    grid_NN_result.fit(x_train_fold, y_train_fold) 
    lst_accu_stratified.append(grid_NN_result.score(x_test_fold, y_test_fold)) 
print('List of possible accuracy:', lst_accu_stratified) 
print('\nMaximum Accuracy That can be obtained from this model is:', 
      max(lst_accu_stratified)*100, '%') 
print('\nMinimum Accuracy:', 
      min(lst_accu_stratified)*100, '%') 
print('\nOverall Accuracy:', 
      mean(lst_accu_stratified)*100, '%')
