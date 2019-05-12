import os
from sklearn.neighbors import KNeighborsClassifier
from cross_validation import cross_validation, sort_attribute
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from tempfile import TemporaryFile

dir_path = os.path.dirname(os.path.realpath(__file__))

def knn(input):    # input = np.random.permutation(input) #losowe wymieszanie wierszy
    outputs = input_normalization(input)
    normal_string = ["without normalization",
                     "with [0,1] min max norm",
                     "with z-score norm",
                     "with max absolute norm"]
    scores = []
    item_iter = 0
    target = np.array(input[:,-1]).reshape(475,1)
    for item in outputs:                                                        # sortujemy cechy znormalizowane według przyjętego jednego formatu klasyfikatora n_neighbours=9, p=2
        sort = np.array(sort_attribute(item, KNeighborsClassifier(n_neighbors=9, p=2, metric='minkowski')))
        scores.append(sort)
    print(scores[1][0][0])
    for item in outputs:
        how_many_attrs = 1                                                      # ile cech chcemy w konkretnym zbiorze danych wykorzystać
        while how_many_attrs <= 30:
            cv_scores = []                                                      # zerujemy wszystko
            attr_index = 0
            column = np.array(item[:, attr_index]).reshape(475, 1)
            data_fill = np.array(column)
            for j in range(0, how_many_attrs):
                attr_index = np.int(scores[item_iter][j][1])                    # iterujemy po cechach o indexie zawartym w tablicy posortowanych cech
                column = np.array(item[:,attr_index]).reshape(475,1)            # reshape do macierzy jednokolumnowej, żeby można było stworzyć macierz cech
                                                                                # kolejne kolumny o indeksach z posortowanej listy cech dodajemy do macierzy
                data_fill = np.hstack((data_fill,column))                       # dodajemy kolumnę do macierzy cech
            full_filled = np.hstack((data_fill, target))                        # łączymy macierz cech z targetem
            for p in range(1, 3):
                for k in range(1,10):
                    scores_mean = (cross_validation(full_filled, KNeighborsClassifier(n_neighbors=k, p=p, metric='minkowski')), # wynik uczenia się
                                                    how_many_attrs,                                                             # dla jakiej liczby cech
                                                    k,                                                                          # dla określonej liczby sąsiadów
                                                    p)                                                                          # i dla określonej metryki
                    cv_scores.append(scores_mean)
            cv_scores = sorted(cv_scores, key = lambda x: x[0],reverse=True)
            how_many_attrs+=1
            print(cv_scores)
            print(full_filled.shape)
        print("ITEM NUMBER %s" %(item_iter+1))
        item_iter+=1
    # fig = 1
    # string_index = 0
    # for item in outputs:
    #     for p in range(1,3):
    #         cv_scores = []
    #         for i in range(1,10):
    #             scores_mean = cross_validation(item, KNeighborsClassifier(n_neighbors=i, p=p, metric='minkowski'))
    #             cv_scores.append(scores_mean)
    #         Range = list(range(1,10))
    #         cv_scores = [x for x in cv_scores]
    #         plt.subplot(4,2,fig)
    #         if p == 1:
    #             plt.title("Manhattan distance measurement "+normal_string[string_index])
    #         if p == 2:
    #             plt.title("Euclidean distance measurement "+normal_string[string_index])
    #         plt.plot(Range, cv_scores)
    #         plt.xlabel('Number of Neighbors K')
    #         plt.ylabel('Learning level')
    #         fig+=1
    #     string_index+=1
    # plt.show()

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
    print(target.shape)
    print("----------")
    print(min_max.shape)
    return no_norm, min_max_out, z_score_out, max_abs_out

