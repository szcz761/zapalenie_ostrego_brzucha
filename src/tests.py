import os
from sklearn.neighbors import KNeighborsClassifier
import helper
import matplotlib.pyplot as plt
from  scipy.stats import wilcoxon
import numpy as np
import time
# dodajemy kolejne cechy w walidacji krzyzowej nie na odwrot dla knn gdzie k = 5
# 1. liczba cech od dokladnosci dla wybranego algorytmu klasyfikacji
# 2. tabelka, testy dla jedenej wybranej liczby cech dla wszystkich liczb sasiadow
dir_path = os.path.dirname(os.path.realpath(__file__))
all_scores=[]
# features = np.arange(5, 32, 1)
# features =[]
# features = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
# features = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43]

def test(input,f,a_, count_):
    global features 
    global a
    global count
    a=a_
    count=count_
    features = f
    start = time.time()
    calculate_and_plot_feature_selection_score(input, helper.random_test, "Losowe odrzucenie cech")
    end = time.time()
    print("RAND: ",end - start)

    start = time.time()
    calculate_and_plot_feature_selection_score(input, helper.SFS_test, "SFS test")
    end = time.time()
    print("SFS: ",end - start)

    start = time.time()
    calculate_and_plot_feature_selection_score(input, helper.wraper_test, "Wraper test")
    end = time.time()
    print("WRAPER: ",end - start)

    start = time.time()
    calculate_and_plot_feature_selection_score(input, helper.PCA_test, "PCA test")
    end = time.time()
    print("PCA: ",end - start)

    start = time.time()
    calculate_and_plot_feature_selection_score(input, helper.pearson_test, "Pearson test")
    end = time.time()
    print("Pearson: ",end - start)
    
    start = time.time()
    calculate_and_plot_feature_selection_score(input, helper.kolmogorov_test, "test Kołmogorowa-Smirnowa")
    end = time.time()
    print("Kolmogorov: ",end - start)
# 
    print("brak normalizacji")
    wilcoxon_for_arrays(all_scores[0],all_scores[2] ,"Random | SFS")
    wilcoxon_for_arrays(all_scores[0],all_scores[4] ,"Random | Wraper")
    wilcoxon_for_arrays(all_scores[0],all_scores[6] ,"Random | PCA")
    wilcoxon_for_arrays(all_scores[0],all_scores[8] ,"Random | Pearson")
    wilcoxon_for_arrays(all_scores[0],all_scores[10],"Random | Kołmogorowa")
    wilcoxon_for_arrays(all_scores[8],all_scores[2] ,"Pearson | SFS")
    wilcoxon_for_arrays(all_scores[8],all_scores[4] ,"Pearson | Wraper")
    wilcoxon_for_arrays(all_scores[8],all_scores[6] ,"Pearson | PCA")
    wilcoxon_for_arrays(all_scores[8],all_scores[10],"Pearson | Kołmogorowa")
    wilcoxon_for_arrays(all_scores[2],all_scores[4] ,"SFS | Wraper")
    wilcoxon_for_arrays(all_scores[2],all_scores[6], "SFS | PCA")
    wilcoxon_for_arrays(all_scores[2],all_scores[10],"SFS | Kołmogorowa")
    wilcoxon_for_arrays(all_scores[4],all_scores[6], "Wraper | PCA")
    wilcoxon_for_arrays(all_scores[4],all_scores[10],"Wraper | Kołmogorowa")
    wilcoxon_for_arrays(all_scores[6],all_scores[10],"PCA | Kołmogorowa")
        
    print("normalizacja")
    wilcoxon_for_arrays(all_scores[1],all_scores[3] ,"Random | SFS")
    wilcoxon_for_arrays(all_scores[1],all_scores[5] ,"Random | Wraper")
    wilcoxon_for_arrays(all_scores[1],all_scores[7] ,"Random | PCA")
    wilcoxon_for_arrays(all_scores[1],all_scores[9] ,"Random | Pearson")
    wilcoxon_for_arrays(all_scores[1],all_scores[11],"Random | Kołmogorowa")
    wilcoxon_for_arrays(all_scores[9],all_scores[3] ,"Pearson | SFS")
    wilcoxon_for_arrays(all_scores[9],all_scores[5] ,"Pearson | Wraper")
    wilcoxon_for_arrays(all_scores[9],all_scores[7] ,"Pearson | PCA")
    wilcoxon_for_arrays(all_scores[9],all_scores[11],"Pearson | Kołmogorowa")
    wilcoxon_for_arrays(all_scores[3],all_scores[5] ,"SFS | Wraper")
    wilcoxon_for_arrays(all_scores[3],all_scores[7], "SFS | PCA")
    wilcoxon_for_arrays(all_scores[3],all_scores[11],"SFS | Kołmogorowa")
    wilcoxon_for_arrays(all_scores[5],all_scores[7], "Wraper | PCA")
    wilcoxon_for_arrays(all_scores[5],all_scores[11],"Wraper | Kołmogorowa")
    wilcoxon_for_arrays(all_scores[7],all_scores[11],"PCA | Kołmogorowa")

    plt.show()

def wilcoxon_for_arrays(array_a,array_b,string) :
    print(string)
    a=[]
    b=[]
    for i in range(len(features)):
        a.append(float(array_a[i]))
        b.append(float(array_b[i]))
    print(wilcoxon(a,b))

def calculate_and_plot_feature_selection_score(input, selection, name):

    outputs = helper.input_normalization(input, a, count)
    fig = plt.figure()
    labels = ["brak normalizacji","normalizacja"]
    i=0
    for item in outputs: #normalizacja i brak
        final = []
        final_2 = []
        cv_scores=[]
        for how_many_attrs in features:#range(1,32):#pierwsze 30 najlepszych cech
            final.append(selection(item,how_many_attrs,cv_scores))

        for item in final:
            final_2.append(helper.cross_validation(item, [KNeighborsClassifier(n_neighbors=5)]))

        # print(final_2)
        all_scores.append(final_2)
        plt.plot(features, final_2, label=labels[i])
        i+=1

    plt.title(name+ " dla algorytmu 5-najbliszych sasiadow")
    plt.legend()
    ax = fig.gca()
    plt.xlabel("Liczba Cech")
    plt.ylabel("Dokladnosc")
    ax.set_xticks(features)
    ax.set_yticks(np.arange(0, 1., 0.1))
    plt.grid()
   

def add_to_list_Classifiers():
    classifiers=[]
    metrics = ["euclidean",
               "manhattan"]
    for metric in metrics:
        for k in [1,5,7,9]:
            classifiers = np.append(classifiers, KNeighborsClassifier(n_neighbors=k, metric=metric))
        # classifiers = np.append(classifiers, NearestCentroid(metric=metric))
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




