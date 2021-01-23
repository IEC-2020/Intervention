## SPLIT THE DATA SET INTO X AND Y###
X = dataframe.drop('Col_to_exclude', axis=1)
y= df_scaled['Col_to_exclude']

## SPLIT into TRAIN and TEST DATA###
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

##LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression(class_weight="balanced")
log_model.fit(X_train, y_train)

### MAKE PREDICTIONS ###
predictions = log_model.predict(X_test)

## CALCULATE THE PREDICTIONS ###
from sklearn.metrics import accuracy_score
print("The accuracy with logistic regression is:", accuracy_score(y_test, predictions))
