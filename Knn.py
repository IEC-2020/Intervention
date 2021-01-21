#### Use Preprocessed dataset ######
####### This is for KNN #################


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=20, stratify=y)
knn = KNeighborsClassifier(3)
knn.fit(X_train,y_train)
print("Train score",knn.score(X_train,y_train),"%")
print("Test score",knn.score(X_test,y_test),"%")
