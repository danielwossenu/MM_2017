from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn import metrics
import numpy as np

import matplotlib.pyplot as plt

import ImplementQuotients

if __name__ == '__main__':
    FE = ImplementQuotients.FeatureEngineering()
    FE.create_features()

    x_train,y_train = FE.GetTrainingData(2012,2014)
    x_test, y_test = FE.GetTestData(2015)

    plt.plot(x_train[y_train==0,0],x_train[y_train==0,3],'bo')
    # plt.plot(x_train[y_train==1,0],x_train[y_train==1,1],'ro')
    # plt.plot(x_train[y_train==1,0],x_train[y_train==1,1],'ro')
    plt.show()

