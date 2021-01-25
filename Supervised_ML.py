A. Take X and Y from Chi2, TREE, Ridge, RFE selected features. 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)
             
B. Scale the features. 
C. Apply these algorithms and add the value of the accuracy in the excel sheet which i given it to you via email. 
D. Add all your code in this sheet.

########################################## Logistic Regression.  ####################################################
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()       Link: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html (Change the value of C)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

##########################################    Naïve Bayes.     ######################################################
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()              Link: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
from sklearn import metrics
print("Accuracy",metrics.accuracy_score(y_test, y_pred))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

########################################## K-Nearest Neighbours. ####################################################
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)       Link : https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html 
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

##########################################   Decision Tree.  ########################################################
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()          Link : https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
####### Decision Tree with entropy use to split the tree because default is Gini index ########
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

##########################################. Random Forest.  #########################################################
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100)    Link: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html 
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
########################################## Stochastic Gradient Descent ##################################################
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
clf =make_pipeline(SGDClassifier())
clf.fit(X_train,y_train)
sgd_pred = clf.predict(X_test)

from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, sgd_pred))
########################################## Support Vector Machine. ##################################################
from sklearn import svm
clf = svm.SVC(kernel='linear')          Link: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
