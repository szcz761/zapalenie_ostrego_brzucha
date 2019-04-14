import pandas as pd
import xlrd
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import sys
from sklearn import datasets
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))

def cross_validation(input):
    # input = np.random.permutation(input) #losowe wymieszanie wierszy
    data = input[:,:31] # dane na podstawie ktorych klayfikujemy (data) - macierz 475x31
    target = input[:,31] # klasy do ktorych klasyfikujemy (target) - kolumna o indeksie 31
    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.2) #nastepuje przelosowanie wierszy
    print("training data shape: %s, training target shape %s" %(train_data.shape, train_target.shape))
    print(test_data.shape, test_target.shape)

    clf = svm.SVC(kernel='linear', C=1)
    scores = cross_val_score(clf, data, target, cv=10)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    #for train, test in KFold(8, True, 1).split(data):
    #    training_data = data[train]
    #    training_target = training_data[:,31]
    #    knearest = KNeighborsClassifier(n_neighbors=3).fit(training_data, training_target)
    #    scores = cross_val_score(knearest, training_data, training_target, cv=10, scoring='accuracy')

    # sprawdzam dla jakiego k najlepiej się zachowuje klasyfikator
    # tzn predykcja wypisuje najmniejszy błąd klasyfikacji
    cv_scores = []
    k=1
    while k<10:
        knn = KNeighborsClassifier(n_neighbors=k).fit(train_data, train_target)
        scores = cross_val_score(knn, train_data, train_target, cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())
        k+=1
    Range = list(range(1,k))
    MSE = [1 - x for x in cv_scores]
    plt.plot(Range, MSE)
    plt.xlabel('Number of Neighbors K')
    plt.ylabel('Misclassification Error')
    plt.show()
    # clf = svm.SVC(kernel='linear', C=1).fit(train_data, train_target)
    # print(clf.score(test_data, test_target))
    # y = np.ravel(y)
    # y = y.flatten()
    # print('##### X #####')
    # print(X.shape)
    # print('##### y #####')
    # print(y.shape)
    # kfold = KFold(2, True, 1)


    # for train, test in kfold.split(X):
    #     #print('train: %s, test: %s' % (X[train], X[test]))
    #     train_X, test_X = X[train], X[test]
    #     train_y, test_y = y[train], y[test]
    # return train_X, test_X, train_y, test_y

def myFunc():
    iris = datasets.load_iris()
    print(iris.data.shape, iris.target.shape)
    print(iris)
    print(iris.data)
    print(iris.target)

    train_data, test_data, train_target, test_target = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)
    print(train_data.shape, train_target.shape)
    print(test_data.shape, test_target.shape)

    clf = svm.SVC(kernel='linear', C=1).fit(train_data, train_target, scoring='f1_macro')
    print(clf.score(test_data, test_target))

if __name__ == "__main__":
    df = pd.read_excel(os.path.join(dir_path, '../data/Stany_ostrego_brzucha-dane.xls'), header=0, skipfooter=15, usecols="A:AF")
    # print(df)
    # np.set_printoptions(threshold=sys.maxsize)
    all_items = df.to_numpy()
    # myFunc()
    cross_validation(all_items)
    # train_X, test_X, train_y, test_y = cross_validation(all_items)
    # print('%s %s %s %s'%(train_X.shape, test_X.shape, train_y.shape, test_y.shape))
    # # No iterations for now
    # # we create an instance of Neighbours Classifier and fit the data,
    # # then print the first prediction.
    # clf = NearestCentroid(shrink_threshold=None)
    # clf.fit(train_X, train_y)
    # y_pred = clf.predict(train_X)
    # print(None, np.mean(train_y == y_pred))