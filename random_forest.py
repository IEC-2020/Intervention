###RANDOM FOREST###
rom sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
rfc=RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
accuracy_score(y_test, rfc_pred)
