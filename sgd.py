### SGD PREDICITON ###
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=1000, tol=1e-3))
clf.fit(X_train, y_train)

### MAKE PREDICTION ###
sgd_prediction = clf.predict(X_test)

### CALCULATE ACCURACY ###
accuracy_score(y_test,sgd_prediction)
