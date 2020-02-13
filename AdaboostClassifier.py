# Load libraries
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
# Import train_test_split function
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
import numpy as np

# Load data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Load data Manual
# data = np.genfromtxt("data_iris.txt",delimiter=",")
# X = []
# y = []
# for i,x in enumerate(data):
#     X.append(x[0:-1])
#     y.append(int(x[-1]))


for j,xa in enumerate(iris.data):
    for i,ya in enumerate(xa):
        print(ya,end=",")
    if(y[j] == 0):
        print('1,0,0')
    elif(y[j] == 1):
        print('0,1,0')
    elif(y[j] == 2):
        print('0,0,1')


# # Split dataset into training set and test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,shuffle=False) # 70% training and 30% test
#
#
#
# # Create adaboost classifer object
#
# # AdaBoost Biasa
# # abc = AdaBoostClassifier(n_estimators=10,learning_rate=1)
#
# # Adaboost SVC
# # svc=SVC(probability=True, kernel='linear')
# # abc =AdaBoostClassifier(n_estimators=50, base_estimator=svc,learning_rate=1)
#
# # Decision Tree
# abc = AdaBoostClassifier(
#     DecisionTreeClassifier(max_depth=1),
#     n_estimators=50
# )
#
#
#
# # Train Adaboost Classifer
# model = abc.fit(X_train, y_train)
#
# #Predict the response for test dataset
# y_pred = model.predict(X_test)
#
# # Model Accuracy, how often is the classifier correct?
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))