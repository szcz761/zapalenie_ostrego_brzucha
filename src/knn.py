import os
from sklearn.neighbors import KNeighborsClassifier
from cross_validation import cross_validation
from sklearn import preprocessing
import matplotlib.pyplot as plt
from numpy import int64

dir_path = os.path.dirname(os.path.realpath(__file__))

def knn(input):    # input = np.random.permutation(input) #losowe wymieszanie wierszy
    outputs = input_normalization(input)
    normal_string = ["without normalization",
                     "with [0,100] min max norm",
                     "with mean norm",
                     "with std norm"]
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

def input_normalization(input): # wyskalowane dane są rzutowane do int64, póki co strata precyzji
    no_norm = input
    min_max = preprocessing.MinMaxScaler(feature_range=(0,100)).fit_transform(input).astype(int64)
    mean = preprocessing.scale(input, with_mean=True).astype(int64)
    std = preprocessing.scale(input, with_std=True).astype(int64)
    return no_norm, min_max, mean, std

