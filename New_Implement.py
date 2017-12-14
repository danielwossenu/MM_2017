from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn import metrics
import numpy as np

import ImplementQuotients

if __name__ == '__main__':
    FE = ImplementQuotients.FeatureEngineering()
    FE.create_features()

    # training_data = FE.GetTrainingData(2003,2014)
    # testing_data = FE.GetTestData(2015)
    #
    # x_train = training_data[0]
    # y_train = training_data[1]
    # x_test = testing_data[0]
    # y_test = testing_data[1]
    #
    #
    # scaler = preprocessing.StandardScaler().fit(x_train)
    # x_train_scaled = scaler.transform(x_train)
    # x_test_scaled = scaler.transform(x_test)
    #
    #
    # model = MLPClassifier(hidden_layer_sizes=(1000,100),verbose=1)
    # model.fit(x_train_scaled,y_train)
    # y_pred = model.predict(x_test_scaled)
    # print(metrics.accuracy_score(y_test,y_pred))


    print 'building models'
    scaler = None
    for x in range(2003, 2015):
        max_n = min((2016 - x) * 64 * 2 / 5, 100)
        x_train, y_train = FE.GetTrainingData(x, 2016)
        scaler = preprocessing.StandardScaler().fit(x_train)
        x_train_scaled = scaler.transform(x_train)
        KNN = KNeighborsClassifier(n_neighbors=40, weights='distance', p=1)
        parameters = {'n_neighbors': range(1, max_n), 'weights': ['uniform', 'distance'], 'p': [1, 2]}
        KNN = model_selection.GridSearchCV(KNN, parameters, n_jobs=-1, cv=5, verbose=1)
        KNN.fit(x_train_scaled, y_train)
        print x
        print KNN.best_score_
        KNN = KNN.best_estimator_
        print KNN