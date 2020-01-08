import os
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
import helper
import matplotlib.pyplot as plt
import numpy as np
# dodajemy kolejne cechy w walidacji krzyzowej nie na odwrot dla knn gdzie k = 5
# 1. liczba cech od dokladnosci dla wybranego algorytmu klasyfikacji
# 2. tabelka, testy dla jedenej wybranej liczby cech dla wszystkich liczb sasiadow
dir_path = os.path.dirname(os.path.realpath(__file__))

def test(input):    # input = np.random.permutation(input) #losowe wymieszanie wierszy
    calculate_and_plot_feature_selection_score(input, helper.pearson_test, "Pearson test")
    calculate_and_plot_feature_selection_score(input, helper.kolmogorov_test, "Kolmogorov test")
    plt.show()
    
def calculate_and_plot_feature_selection_score(input, selection, name):
    outputs = helper.input_normalization(input)
    fig = plt.figure()
    for item in outputs: #normalizacja i brak
        target = np.array(item[:,-1]).reshape(475,1)
        scores = np.array(selection(item))
        final = []
        final_2 = []
        for how_many_attrs in range(1,32):#pierwsze 30 najlepszych cech
            for attrs_iter in range(-1,how_many_attrs):
                data_fill = helper.adding_attribute(attrs_iter, item, scores)
                full_filled = np.hstack((data_fill, target))
            final.append(full_filled)
        for item in final:
            final_2.append(helper.cross_validation(item, [KNeighborsClassifier(n_neighbors=5, metric="euclidean")]))
        if type(item[0][0]) == np.float64:
            label = "normaalizacja"
        else:
            label = "normaalizacja brak"
        print(final_2)
        plt.plot(np.arange(1, 32, 1), final_2, label=label)

    plt.title(name+ " feature for knn 5 neighbors euclidean")
    plt.legend()
    ax = fig.gca()
    ax.set_xticks(np.arange(1, 32, 1))
    ax.set_yticks(np.arange(0, 1., 0.1))
    plt.grid()
   

def add_to_list_Classifiers():
    classifiers=[]
    metrics = ["euclidean",
               "manhattan"]
    for metric in metrics:
        for k in [1,5,7,9]:
            classifiers = np.append(classifiers, KNeighborsClassifier(n_neighbors=k, metric=metric))
        classifiers = np.append(classifiers, NearestCentroid(metric=metric))
    return classifiers



def k_choice(input):
    fig = 1
    string_index = 0
    k_chosen = []
    normal_string = []
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




