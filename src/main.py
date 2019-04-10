import pandas as pd
import xlrd
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestCentroid
import sys
from sklearn import datasets
from sklearn import svm

dir_path = os.path.dirname(os.path.realpath(__file__))


def get_matrix_from_vector(data, vector):
    result = []
    for row in data:
        if row[0] in vector
            result.append(row)

    return result
def cross_validation(input):
    # input = np.random.permutation(input) #losowe wymieszanie wierszy
    
    # print(train_data.shape, train_target.shape)
    # print(test_data.shape, test_target.shape)

    print("Test KFold")
    kf = KFold(n_splits=5)  # na ile podzbiorw dzielimy 20% to test gdy jest n_split = 5
    print(kf)
    for train, test in kf.split(input): #ZWRACA WEKTORY A NIE MACIERZE!!!!!!!
        train_data = input[train[0]:train[-1],:31] # dane na podstawie ktorych klayfikujemy (data) - macierz 475x31
        test_data = input[test[0]:test[-1],:31]
        train_target = input[train[0]:train[-1]:,31] # klasy do ktorych klasyfikujemy (target) - kolumna o indeksie 31
        test_target = input[test[0]:test[-1]:,31]
        clf = svm.SVC(kernel='linear', C=1).fit(train_data, train_target)
        # scores = cross_val_score(clf, data, target, cv=10)
        print(clf.score(test_data, test_target))
        # print(scores)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    # clf = svm.SVC(kernel='linear', C=1).fit(train_data, train_target)
    # print(clf.score(test_data, test_target))
    # y = np.ravel(y)
    # y = y.flatten()
    # print('##### X #####')
    # print(X.shape)
    # print('##### y #####')
    # print(y.shape)
    # kfold = KFold(2, True, 1)

    # #Works for y set to size of 31 not 475 yet

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
    df = pd.read_excel(os.path.join(dir_path, '../data/Stany_ostrego_brzucha-dane.xls'), header=0, skip_footer=15, parse_cols="A:AF")
    print(df)
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