# Load libraries
from sklearn.ensemble import AdaBoostClassifier
# Import train_test_split function
from sklearn.tree import DecisionTreeClassifier
#Import scikit-learn metrics module for accuracy calculation
import numpy as np, os , time
from sklearn.model_selection import KFold
from validate import validation
from sklearn.tree.export import export_text

from sklearn import tree

def enumarate_file(directory):
    return [filename for filename in sorted(os.listdir(directory))]

def prediction(x,y):
    hasil = []
    for index,xx in enumerate(x):
        # print(str(x[index])+" : "+str(y[index]))
        if(x[index] == y[index]):
            hasil.append(1)
        else:
            hasil.append(0)
    return hasil


if __name__ == '__main__':
    #Init
    # kelas = 3
    # data_awal = "data_iris_3.txt"

    # kelas = 5
    # # data_awal = "data_awal"
    # data_awal = "../data_2017-2019_5-label"
    # data_hasil = "data_hasil-5-label"

    kelas = 10
    # data_awal = "data_awal"
    data_awal = "../data_2017-2019_10-label"
    data_hasil = "data_hasil-10-label"


    # COBA
    # kelas = 5
    # data_awal = "DATACOBA/data_2017-2019_5-label"
    # data_hasil = "DATACOBA/data_hasil_5-label"

    # NEW kelas = 5
    # # data_awal = "data_awal"
    # kelas = 5
    # data_awal = "../new_data_2015-2019_5-label"
    # data_hasil = "new_data_hasil_5-label"

    max_depths = [1]
    n_estimators = [60,70,80,90,100]

    for max_depth in max_depths:
        for n_estimator in n_estimators:
            for file in enumarate_file(data_awal):
                subfolder = str(max_depth) + "-depth_" + str(n_estimator) + "-estimator"
                print(data_hasil + "/" + subfolder)
                if not os.path.exists(data_hasil + "/" + subfolder):
                    os.makedirs(data_hasil + "/" + subfolder)

                new_file = open(data_hasil + "/" + subfolder + "/" + subfolder + "_" + file, 'w')

                # Load data Manual
                data = np.genfromtxt(data_awal + "/" + file, delimiter=",")
                hasil_predict = []

                X = []
                for i, x in enumerate(data):
                    X.append(x[0:-kelas])

                Y = []
                for l in range(1, kelas + 1):
                    y = []
                    for i, x in enumerate(data):
                        y.append(int(x[-l]))
                    Y.append(y)

                # FOLD
                # kf = KFold(n_splits=10,random_state=None, shuffle=True)
                kf = KFold(n_splits=10, shuffle=True)
                for train_index, test_index in kf.split(X):
                    start = time.process_time()
                    pred = []
                    test = []
                    for l in range(0, kelas):
                        # Split dataset into training set and test set
                        X_train, X_test = X[train_index[0]:train_index[-1]], X[test_index[0]:test_index[-1]]
                        y_train, y_test = Y[l][train_index[0]:train_index[-1]], Y[l][test_index[0]:test_index[-1]]

                        # Create adaboost classifer object

                        # AdaBoost Biasa
                        # abc = AdaBoostClassifier(n_estimators=10,learning_rate=1)

                        # Adaboost SVC
                        # svc=SVC(probability=True, kernel='linear')
                        # abc =AdaBoostClassifier(n_estimators=50, base_estimator=svc,learning_rate=1)

                        # Decision Tree
                        abc = AdaBoostClassifier(
                            DecisionTreeClassifier(max_depth=max_depth),
                            n_estimators=n_estimator
                        )

                        # print('%%%%%%%%%%%%%%%%')

                        # Train Adaboost Classifer
                        model = abc.fit(X_train, y_train)

                        # decision_tree = DecisionTreeClassifier(max_depth=max_depth)
                        # decision_tree = decision_tree.fit(X_train, y_train)
                        # r = export_text(decision_tree)


                        # Predict the response for test dataset
                        y_pred = model.predict(X_test)

                        pred.append(y_pred)
                        test.append(y_test)

                    predT = np.array(pred).T.tolist()
                    testT = np.array(test).T.tolist()
                    # print(predT) # Prediction
                    # print('%%%%%%%%%%%%%%%%')
                    # print(testT) # Test
                    validation_result = validation(predT, testT)

                    end = time.process_time()
                    learning_time = str(end - start)
                    print(validation_result + "|" + learning_time)
                    new_file.write(validation_result + "|" + learning_time + "\n")

                new_file.close()

