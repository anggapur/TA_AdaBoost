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
from sklearn.model_selection import KFold

def prediction(x,y):
    hasil = []
    for index,xx in enumerate(x):
        # print(str(x[index])+" : "+str(y[index]))
        if(x[index] == y[index]):
            hasil.append(1)
        else:
            hasil.append(0)
    return hasil

#Init
kelas = 2

# Load data Manual
data = np.genfromtxt("data_iris_2.txt",delimiter=",")
hasil_predict = []

for l in range(1,kelas+1):
    hp = []
    X = []
    y = []
    for i,x in enumerate(data):
        X.append(x[0:-kelas])
        y.append(int(x[-l]))

    # FOLD
    # kf = KFold(n_splits=10,random_state=None, shuffle=True)
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(X):

        # Split dataset into training set and test set
        X_train, X_test = X[train_index[0]:train_index[-1]], X[test_index[0]:test_index[-1]]
        y_train, y_test = y[train_index[0]:train_index[-1]], y[test_index[0]:test_index[-1]]


        # Create adaboost classifer object

        # AdaBoost Biasa
        # abc = AdaBoostClassifier(n_estimators=10,learning_rate=1)

        # Adaboost SVC
        # svc=SVC(probability=True, kernel='linear')
        # abc =AdaBoostClassifier(n_estimators=50, base_estimator=svc,learning_rate=1)

        # Decision Tree
        abc = AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=1),
            n_estimators=50
        )


        # Train Adaboost Classifer
        model = abc.fit(X_train, y_train)

        #Predict the response for test dataset
        y_pred = model.predict(X_test)

        # print(prediction(y_test,y_pred))
        hp.append(prediction(y_test,y_pred))
        # Model Accuracy, how often is the classifier correct?
        # print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    hasil_predict.append(hp)

# Coba View
for ix,x in enumerate(hasil_predict):
    for iy, y in enumerate(x):
        print(y)
    print('---')

d1 = len(hasil_predict)
d2 = len(hasil_predict[0])

print('----------')
print('----------')

for ix in range(0,d2):
    datas = []
    for iy in range(0,d1):
        datas.append(hasil_predict[iy][ix])
    print(datas)
    print("---------------")