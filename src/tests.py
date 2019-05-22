import os
from sklearn.neighbors import KNeighborsClassifier
import helper
import matplotlib.pyplot as plt
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))

def knn(input):    # input = np.random.permutation(input) #losowe wymieszanie wierszy
    outputs = helper.input_normalization(input)
    normal_string = ["without normalization",
                     "with z-score norm"]
    scores = []
    item_iter = 0
    target = np.array(input[:,-1]).reshape(475,1)                                                      # sortujemy wtórnie cechy znormalizowane według przyjętego jednego formatu klasyfikatora, np. n_neighbours=9, p=1
    scores = np.array(helper.kolmogorov_test(input))
    # print(scores[0][0])
    for item in outputs:
        # print("ITEM " +normal_string[item_iter])
        for how_many_attrs in range(1,31):
            cv_scores, data_fill = helper.adding_attribute(how_many_attrs, item, item_iter, scores)
            full_filled = np.hstack((data_fill, target))                        # łączymy macierz cech z targetem
            classifiers = create_lists_KNeighborsClassifiers()
            helper.cross_validation(full_filled, classifiers)
            # print(cv_scores)
            # print(full_filled.shape)
        item_iter+=1

def create_lists_KNeighborsClassifiers():
    classifiers=[]
    for p in [1, 2]:
        for k in [1,5,7,9]:
            classifiers = np.append(classifiers, KNeighborsClassifier(n_neighbors=k, p=p, metric='minkowski'))
    return classifiers





def k_choice(input):
    fig = 1
    string_index = 0
    k_chosen = []
    k_final = []
    scores = []
    for p in [1, 2]:
        cv_scores = []
        for i in range(1, 10):
            scores_mean = (helper.cross_validation(input, KNeighborsClassifier(n_neighbors=i, p=p, metric='minkowski')),i)
            cv_scores.append(scores_mean)
            scores.append(scores_mean)
        Range = list(range(1, 10))
        cv_scores = [x[0] for x in cv_scores]
        plt.subplot(4, 2, fig)
        if p == 1:
            plt.title("Manhattan distance measurement " + normal_string[string_index])
        if p == 2:
            plt.title("Euclidean distance measurement " + normal_string[string_index])
        plt.plot(Range, cv_scores)
        plt.xlabel('Number of Neighbors K')
        plt.ylabel('Learning level')
        fig += 1
        string_index += 1
    final_scores = sorted(scores, key=lambda x: x[0], reverse=True)
    k_chosen.append(final_scores)
    for i in range(4):
        print(k_chosen[0][i][1])
    plt.show()




