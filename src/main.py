# from sklearn import svm
import pandas as pd
import numpy as np
import os
import helper
from knn import knn

dir_path = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    df = pd.read_excel(os.path.join(dir_path, '../data/Stany_ostrego_brzucha-dane.xls'),
                       header=0, skipfooter=15, use_cols="A:AG")
    #print(df)
    knn(df.to_numpy())

    # cross_validation(all_items, svm.SVC(kernel='linear', C=1))
