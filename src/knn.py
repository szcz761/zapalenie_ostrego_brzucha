import os
from sklearn.neighbors import KNeighborsClassifier
import helper

import matplotlib.pyplot as plt
import numpy as np
from tempfile import TemporaryFile

dir_path = os.path.dirname(os.path.realpath(__file__))

def knn(input):    # input = np.random.permutation(input) #losowe wymieszanie wierszy
    outputs = helper.input_normalization(input)
    normal_string = ["without normalization",
                     "with [0,1] min max norm",
                     "with z-score norm",
                     "with max absolute norm"]
    scores = []
    item_iter = 0
    target = np.array(input[:,-1]).reshape(475,1)
    for item in outputs:                                                        # sortujemy cechy znormalizowane według przyjętego jednego formatu klasyfikatora n_neighbours=9, p=2
        sort = np.array(helper.sort_attribute(item, KNeighborsClassifier(n_neighbors=9, p=2, metric='minkowski')))
        scores.append(sort)
    print(scores[1][0][0])
    for item in outputs:
        print("ITEM " +normal_string[item_iter])
        for how_many_attrs in range(1,31):
            cv_scores = []                                                      # zerujemy wszystko
            attribute_index = 0
            column = np.array(item[:, attribute_index]).reshape(475, 1)
            data_fill = np.array(column)
            for j in range(0, how_many_attrs):
                attribute_index = np.int(scores[item_iter][j][1])                    # iterujemy po cechach o indexie zawartym w tablicy posortowanych cech
                column = np.array(item[:,attribute_index]).reshape(475,1)            # reshape do macierzy jednokolumnowej, żeby można było stworzyć macierz cech
                                                                                # kolejne kolumny o indeksach z posortowanej listy cech dodajemy do macierzy
                data_fill = np.hstack((data_fill,column))                       # dodajemy kolumnę do macierzy cech
            full_filled = np.hstack((data_fill, target))                        # łączymy macierz cech z targetem
            for p in [1,2]:
                for k in range(1,10):
                    scores_mean = (helper.cross_validation(full_filled, KNeighborsClassifier(n_neighbors=k, p=p, metric='minkowski')), # wynik uczenia się
                                                    how_many_attrs,                                                             # dla jakiej liczby cech
                                                    k,                                                                          # dla określonej liczby sąsiadów
                                                    p)                                                                          # i dla określonej metryki
                    cv_scores.append(scores_mean)
            cv_scores = sorted(cv_scores, key = lambda x: x[0],reverse=True)
            print(cv_scores)
            print(full_filled.shape)
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



