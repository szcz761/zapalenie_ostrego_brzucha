import pandas as pd
import numpy as np
import os
import helper
from tests import test
import helper
from sklearn.neighbors import KNeighborsClassifier

dir_path = os.path.dirname(os.path.realpath(__file__))



if __name__ == "__main__":
    df = pd.read_excel(os.path.join(dir_path, '../data/meaterD.xls'),header=0, skipfooter=15)
    features = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43]
    a=43 
    count =164
    test(df.to_numpy(),features, a,count)

    df = pd.read_excel(os.path.join(dir_path, '../data/Stany_ostrego_brzucha-dane.xls'),header=0, skipfooter=15, use_cols="A:AG")
    features = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31] 
    a=30
    count =475
    test(df.to_numpy(),features, a,count)
