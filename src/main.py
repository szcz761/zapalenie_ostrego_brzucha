import pandas as pd
import xlrd
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestCentroid

dir_path = os.path.dirname(os.path.realpath(__file__))

def cross_validation(input):
    split = 31
    input = np.transpose(input)
    input = np.random.permutation(input)
    X,y = input[:split,:], input[split:,:]
    y = np.ravel(y)
    y = y.flatten()
    print('##### X #####')
    print(X.shape)
    print('##### y #####')
    print(y.shape)
    kfold = KFold(2, True, 1)

    #Works for y set to size of 31 not 475 yet

    for train, test in kfold.split(X):
        #print('train: %s, test: %s' % (X[train], X[test]))
        train_X, test_X = X[train], X[test]
        train_y, test_y = y[train], y[test]
    return train_X, test_X, train_y, test_y

if __name__ == "__main__":
    df = pd.read_excel(os.path.join(dir_path, '../data/Stany_ostrego_brzucha-dane.xls'), header=0, skip_footer=15, parse_cols="A:AF")
    print(df)
    all_items = df.to_numpy()

    train_X, test_X, train_y, test_y = cross_validation(all_items)
    print('%s %s %s %s'%(train_X.shape, test_X.shape, train_y.shape, test_y.shape))
    # No iterations for now
    # we create an instance of Neighbours Classifier and fit the data,
    # then print the first prediction.
    clf = NearestCentroid(shrink_threshold=None)
    clf.fit(train_X, train_y)
    y_pred = clf.predict(train_X)
    print(None, np.mean(train_y == y_pred))