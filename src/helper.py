import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from scipy import stats


def cross_validation(input, classifier):
    result_array = np.empty((0, 10))
#     print("Cross Validation")

    input = np.random.permutation(input)  # losowe wymieszanie wierszy
    X, y = input[:, :-1], input[:, -1]
    #print(np.unique(y, return_counts=True))

    # na ile podzbiorw dzielimy 20% to test gdy jest n_split = 5, KFold Stratyfikowany
    # czyli dzielimy podzbiory jednak w sposób inny niż dla problemu binarnego(more accuracy)
    kf = StratifiedKFold(n_splits=5, shuffle=False)
    # ZWRACA WEKTORY A NIE MACIERZE!!!!!!!
    for train, test in kf.split(X, y):
        # X -  dane na podstawie ktorych klayfikujemy - macierz 475x31
        # y -  klasy do ktorych klasyfikujemy (target) - kolumna o indeksie 31
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        # trening
        clf = classifier.fit(X_train, y_train)
        # test
        # if(i % 2 == 0):
        #     sample_pred = clf.predict(test_data)
        #     print("Przykładowa predykcja co którąś iteracje algorytmu:")
        #     print(sample_pred)
        #     print(sample_pred.shape)
        #     print("Porównanie do testowego targetu")
        #     print(test_target)
        #     print(test_target.shape)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        bac = balanced_accuracy_score(y_test, y_pred)

        single_result = clf.score(X_test, y_test)

        print("SR %.3f | %.3f | bac %.3f" % (
            single_result, accuracy, bac
        ))
        #exit()
        result_array = np.append(result_array, single_result)
#     print(result_array)
#     print("Srednia:")

    return np.mean(result_array)

def kolmogorov_test(input):
    cv_scores = []
    target = np.array(input[:, -1])
    for i in range(0,30):
        one_attribute_data = input[:, i]
        # wykonujemy test kolgomorowa dla każdej cechy względem klas
        # co outputuje zależność statystyczną jednego obiektu od drugiego
        scores_mean = (stats.ks_2samp(target, one_attribute_data),i)
        cv_scores.append(scores_mean)
    cv_scores = sorted(cv_scores, key = lambda x: x[0],reverse=True)
    print(cv_scores)
    return cv_scores

def input_normalization(input):
    data = input[:,:-1] # skalowane są tylko dane, nie target
    target = np.array(input[:,-1]).reshape(475,1) # target przekształcany jest na macierz jednokolumnową
    no_norm = input
    min_max = preprocessing.MinMaxScaler(feature_range=(0,1)).fit_transform(data) # skalowanie min-max
    z_score = preprocessing.StandardScaler(with_mean=True, with_std=True).fit_transform(data) #skalowanie z-score
    max_abs = preprocessing.MaxAbsScaler().fit_transform(data) #skalowanie max abs, czyli przedziały modułu maksymalnych wartości
    min_max_out = np.hstack((min_max, target)) # przeskalowane dane są łączone z targetem
    z_score_out = np.hstack((z_score, target))
    max_abs_out = np.hstack((max_abs, target))
    # print(target.shape)
    # print("----------")
    # print(min_max.shape)
    return no_norm, min_max_out, z_score_out, max_abs_out


def adding_attribute(how_many_attrs, item, item_iter, scores):
    cv_scores = []  # zerujemy wszystko
    attribute_index = 0
    column = np.array(item[:, attribute_index]).reshape(475, 1)
    data_fill = np.array(column)
    for j in range(0, how_many_attrs):
        attribute_index = np.int(
            scores[item_iter][j][1])  # iterujemy po cechach o indexie zawartym w tablicy posortowanych cech
        column = np.array(item[:, attribute_index]).reshape(475,
                                                            1)  # reshape do macierzy jednokolumnowej, żeby można było stworzyć macierz cech
        # kolejne kolumny o indeksach z posortowanej listy cech dodajemy do macierzy
        data_fill = np.hstack((data_fill, column))  # dodajemy kolumnę do macierzy cech
    return cv_scores, data_fill


