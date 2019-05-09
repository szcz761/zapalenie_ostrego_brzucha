from sklearn.model_selection import KFold
from sklearn import svm
import pandas as pd
import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))


def cross_validation(input):
    result_array = np.empty((0,10))
    print("Cross Validation")
    for i in range(0,5):
        input = np.random.permutation(input) #losowe wymieszanie wierszy
        kf = KFold(n_splits=2,shuffle=False)  # na ile podzbiorw dzielimy 20% to test gdy jest n_split = 5
        for vector_of_train_index, vector_of_test_index in kf.split(input): #ZWRACA WEKTORY A NIE MACIERZE!!!!!!!

            train_data = input[vector_of_train_index,:31] # dane na podstawie ktorych klayfikujemy (data) - macierz 475x31
            test_data = input[vector_of_test_index,:31]
            train_target = input[vector_of_train_index,31] # klasy do ktorych klasyfikujemy (target) - kolumna o indeksie 31
            test_target = input[vector_of_test_index,31]

            clf = svm.SVC(kernel='linear', C=1).fit(train_data, train_target)
            single_result = clf.score(test_data, test_target)
            result_array = np.append(result_array,single_result)
    print (result_array)
    print("Srednia:")
    print(np.mean(result_array))


if __name__ == "__main__":
    df = pd.read_excel(os.path.join(dir_path, '../data/Stany_ostrego_brzucha-dane.xls'), header=0, skip_footer=15, parse_cols="A:AG")
    print(df)

    all_items = df.to_numpy()
    cross_validation(all_items)