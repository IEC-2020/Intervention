#####################################################################################################
###################################################### Feature Importance ###########################
#####################################################################################################

from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.datasets import make_friedman1
from sklearn.svm import SVR
import pandas as pd


#### Load the dataset####
#Load the Dataset#
dataset = pd.read_excel('02_Stratified_imputated_data.xlsx')

#Split the dataset#
Y= dataset.iloc[:, 0:1].values
X= dataset.iloc[:, 1:33].values
X=pd.DataFrame(data=X)


#### Ch2 square method ####
X_norm = MinMaxScaler().fit_transform(X)
X_norm=pd.DataFrame(data=X_norm)
chi_selector = SelectKBest(chi2, k=10)
chi_selector.fit(X_norm, Y)
chi_support = chi_selector.get_support()
chi_features = X.loc[:, chi_support].columns.tolist()
print(str(len(chi_features)), 'selected features')
score_chi_selector= chi_selector.scores_


#### RFE #####
X_norm, Y = make_friedman1(n_samples=83, n_features=35, random_state=0)
estimator = SVR(kernel="linear")
selector = RFE(estimator, n_features_to_select=20, step=1)
selector = selector.fit(X_norm, Y)
selector.support_
selector.ranking_


##### Ridge algorithm ####
ridge = Ridge().fit(X_norm, Y)
model = SelectFromModel(ridge, prefit=True, threshold='mean')
X_transformed = model.transform(X_norm)
