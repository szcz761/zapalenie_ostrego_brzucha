
import numpy as np
from sklearn.model_selection import KFold


def cross_validation(input, classifier):
    result_array = np.empty((0, 10))
    print("Cross Validation")
    for i in range(0, 5):
        input = np.random.permutation(input)  # losowe wymieszanie wierszy
        # na ile podzbiorw dzielimy 20% to test gdy jest n_split = 5
        kf = KFold(n_splits=2, shuffle=False)
        # ZWRACA WEKTORY A NIE MACIERZE!!!!!!!
        for vector_of_train_index, vector_of_test_index in kf.split(input):

            # dane na podstawie ktorych klayfikujemy (data) - macierz 475x31
            train_data = input[vector_of_train_index, :31]
            test_data = input[vector_of_test_index, :31]
            # klasy do ktorych klasyfikujemy (target) - kolumna o indeksie 31
            train_target = input[vector_of_train_index, 31]
            test_target = input[vector_of_test_index, 31]

            clf = classifier.fit(train_data, train_target)
            single_result = clf.score(test_data, test_target)
            result_array = np.append(result_array, single_result)
    print(result_array)
    print("Srednia:")
    print(np.mean(result_array))
