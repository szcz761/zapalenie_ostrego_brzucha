
import numpy as np
from sklearn.model_selection import KFold


def cross_validation(input, classifier):
    result_array = np.empty((0, 10))
#     print("Cross Validation")
    for i in range(0, 5):
        input = np.random.permutation(input)  # losowe wymieszanie wierszy
        # na ile podzbiorw dzielimy 20% to test gdy jest n_split = 5
        kf = KFold(n_splits=2, shuffle=False)
        # ZWRACA WEKTORY A NIE MACIERZE!!!!!!!
        for vector_of_train_index, vector_of_test_index in kf.split(input):

            # dane na podstawie ktorych klayfikujemy (data) - macierz 475x31
            train_data = input[vector_of_train_index, :-1]
            test_data = input[vector_of_test_index, :-1]
            # klasy do ktorych klasyfikujemy (target) - kolumna o indeksie 31
            train_target = input[vector_of_train_index, -1]
            test_target = input[vector_of_test_index, -1]

            clf = classifier.fit(train_data, train_target)
            single_result = clf.score(test_data, test_target)
            result_array = np.append(result_array, single_result)
#     print(result_array)
#     print("Srednia:")
#     print(np.mean(result_array))
    return np.mean(result_array)

def sort_attribute(input, classifier):
    cv_scores = []
    for i in range(0,30):
        one_attribute_data = input[:,[i,31]]
        scores_mean = (cross_validation(one_attribute_data, classifier),i)
        cv_scores.append(scores_mean)
    cv_scores = sorted(cv_scores, key = lambda x: x[0],reverse=True)
    print(cv_scores)
    return cv_scores

# def adding_attribute(input, classifier):
#     cv_scores = []
#     sort_attribute(input, classifier)
#     for i in range(1,10):
#         scores_mean = cross_validation(input, KNeighborsClassifier(n_neighbors=i))
#         cv_scores.append(scores_mean)

#     Range = list(range(1,10))
#     MSE = [ x for x in cv_scores]
#     plt.plot(Range, MSE)
#     plt.xlabel('Number of Neighbors K')
#     plt.ylabel('Learning level')
#     plt.show()
