from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.neighbors import NearestCentroid
from sklearn import datasets, svm
import pandas as pd
import numpy as np
import xlrd
import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))


def cross_validation(input):
    input = np.random.permutation(input) #losowe wymieszanie wierszy

    print("Test KFold")
    kf = KFold(n_splits=5,shuffle=False)  # na ile podzbiorw dzielimy 20% to test gdy jest n_split = 5
    print(kf)
    res = np.empty((0,5))
    for vector_of_train_index, vector_of_test_index in kf.split(input): #ZWRACA WEKTORY A NIE MACIERZE!!!!!!!

        train_data = input[vector_of_train_index,:31] # dane na podstawie ktorych klayfikujemy (data) - macierz 475x31
        test_data = input[vector_of_test_index,:31]
        train_target = input[vector_of_train_index,31] # klasy do ktorych klasyfikujemy (target) - kolumna o indeksie 31
        test_target = input[vector_of_test_index,31]

        clf = svm.SVC(kernel='linear', C=1).fit(train_data, train_target)
        NAN = clf.score(test_data, test_target)
        res = np.append(res,NAN)
        print(NAN)
    print("Srednia:")
    print(np.mean(res))


if __name__ == "__main__":
    df = pd.read_excel(os.path.join(dir_path, '../data/Stany_ostrego_brzucha-dane.xls'), header=0, skip_footer=15, parse_cols="A:AG")
    print(df)

    all_items = df.to_numpy()
    cross_validation(all_items)