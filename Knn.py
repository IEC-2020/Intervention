#### Use Preprocessed dataset ######
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=20, stratify=y)

####### This is for KNN #################
from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier()
knn.fit(X_train, y_train)

pred_KNN = knn.predict(X_test)
accuracy_score(y_test, pred_KNN)


print("Train score",knn.score(X_train,y_train),"%")
print("Test score",knn.score(X_test,y_test),"%")

### PLOT TO SHOW ERROR RATE###
error_rate=[]
for i in range(1,40):
  knn = KNeighborsClassifier(n_neighbors=i)
  knn.fit(X_train, y_train)
  pred_i = knn.predict(X_test)
  error_rate.append(np.mean(pred_i!=y_test))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40), error_rate,color="blue")
plt.title("Error Rate vs KNN")
