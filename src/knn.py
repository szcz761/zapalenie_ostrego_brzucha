import os
from sklearn.neighbors import KNeighborsClassifier
from cross_validation import cross_validation
from sklearn import preprocessing
import matplotlib.pyplot as plt
from numpy import int64
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))

def knn(input):    # input = np.random.permutation(input) #losowe wymieszanie wierszy
    outputs = input_normalization(input)
    normal_string = ["without normalization",
                     "with [0,1] min max norm",
                     "with z-score norm",
                     "with max absolute norm"]

    fig = 1
    string_index = 0
    for item in outputs:
        for p in range(1,3):
            cv_scores = []
            for i in range(1,10):
                scores_mean = cross_validation(item, KNeighborsClassifier(n_neighbors=i, p=p, metric='minkowski'))
                cv_scores.append(scores_mean)
            Range = list(range(1,10))
            cv_scores = [x for x in cv_scores]
            plt.subplot(4,2,fig)
            if p == 1:
                plt.title("Manhattan distance measurement "+normal_string[string_index])
            if p == 2:
                plt.title("Euclidean distance measurement "+normal_string[string_index])
            plt.plot(Range, cv_scores)
            plt.xlabel('Number of Neighbors K')
            plt.ylabel('Learning level')
            fig+=1
        string_index+=1
    plt.show()

def input_normalization(input):
    data = input[:,:-1] # skalowane są tylko dane, nie target
    target = np.array(input[:,-1]).reshape(475,1) # target przekształcany jest na macierz jednokolumnową
    no_norm = input
    min_max = preprocessing.MinMaxScaler(feature_range=(0,1)).fit_transform(data) # skalowanie min-max
    z_score = preprocessing.StandardScaler(with_mean=True, with_std=True).fit_transform(data) #
    max_abs = preprocessing.MaxAbsScaler().fit_transform(data)
    min_max_out = np.hstack((min_max, target)) # przeskalowane dane są łączone z targetem
    z_score_out = np.hstack((z_score, target))
    max_abs_out = np.hstack((max_abs, target))
    print(target.shape)
    print("----------")
    print(min_max.shape)
    return no_norm, min_max_out, z_score_out, max_abs_out

