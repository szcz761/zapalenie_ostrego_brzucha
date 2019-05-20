import pandas as pd
import numpy as np
import os
import helper
from knn import knn
import helper
from sklearn.neighbors import KNeighborsClassifier

dir_path = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    df = pd.read_excel(os.path.join(dir_path, '../data/Stany_ostrego_brzucha-dane.xls'),
                       header=0, skipfooter=15, use_cols="A:AG")
    #print(df)
    knn(df.to_numpy())

    #helper.cross_validation(df.to_numpy(), KNeighborsClassifier(n_neighbors=9, p=1, metric='minkowski'))
    #helper.kolgomorov_test(df.to_numpy())