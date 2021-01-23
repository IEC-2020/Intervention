A. Take X and Y from Chi2, TREE, Ridge, RFE selected features. 

                from sklearn.model_selection import train_test_split
                x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)
             
B. Scale the features. 
C. Apply these algorithms and add the value of the accuracy in the excel sheet which i given it to you via email. 
D. Add all your code in this sheet.

########################################## Logistic Regression.  ####################################################
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)

##########################################    Naïve Bayes.     ######################################################
########################################## Stochastic Gradient Descent. #############################################
########################################## K-Nearest Neighbours. ####################################################
##########################################   Decision Tree.  ########################################################
##########################################. Random Forest.  #########################################################
########################################## Support Vector Machine. ##################################################
