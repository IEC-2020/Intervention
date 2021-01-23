### DATA PREPROCESSING ###
### DATA FROM INTERVENTION ###
### Only Selected FEATURES ###
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

### SVM ###
model = SVC()
model.fit(X_train, y_train)

### PREDICTIONS ###
svm_pred = model.predict(X_test)

### ACCURACY ###
print(accuracy_score(y_test,dt_pred))

##REPORT##
print(confusion_matrix(y_test,svm_pred)
print("\n")
print(classification_report(y_test,svm_pred)

