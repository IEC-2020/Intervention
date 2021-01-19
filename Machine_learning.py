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
