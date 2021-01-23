###DECISION TREE###
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

dt_pred = clf.predict(X_test)
cross_val_score(clf,X_train,y_train, cv=100)
 
from sklearn.metrics import accuracy_score
accuracy_score(y_test,dt_pred)
