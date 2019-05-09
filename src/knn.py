import os
from sklearn.neighbors import KNeighborsClassifier
from cross_validation import cross_validation
from sklearn import preprocessing
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))

def knn(input):    # input = np.random.permutation(input) #losowe wymieszanie wierszy

    scaled_input = preprocessing.scale(input)
    cv_scores = []
    for i in range(1,10):
        scores_mean = cross_validation(input, KNeighborsClassifier(n_neighbors=i, p=1, metric='minkowski'))
        cv_scores.append(scores_mean)

    Range = list(range(1,10))
    MSE = [ x for x in cv_scores]
    plt.figure(1)
    plt.title("Manhattan distance measurement without normalization")
    plt.plot(Range, MSE)
    plt.xlabel('Number of Neighbors K')
    plt.ylabel('Learning level')


    cv_scores = []
    for i in range(1,10):
        scores_mean = cross_validation(input, KNeighborsClassifier(n_neighbors=i, p=2, metric='minkowski'))
        cv_scores.append(scores_mean)

    Range = list(range(1,10))
    MSE = [ x for x in cv_scores]
    plt.figure(2)

    plt.title("Euclidean distance measurement without normalization")
    plt.plot(Range, MSE)
    plt.xlabel('Number of Neighbors K')
    plt.ylabel('Learning level')

    cv_scores = []
    for i in range(1,10):
        scores_mean = cross_validation(scaled_input, KNeighborsClassifier(n_neighbors=i, p=1, metric='minkowski'))
        cv_scores.append(scores_mean)

    Range = list(range(1,10))
    MSE = [ x for x in cv_scores]
    plt.figure(3)
    plt.title("Manhattan distance measurement with normalization")
    plt.plot(Range, MSE)
    plt.xlabel('Number of Neighbors K')
    plt.ylabel('Learning level')


    cv_scores = []
    for i in range(1,10):
        scores_mean = cross_validation(scaled_input, KNeighborsClassifier(n_neighbors=i, p=2, metric='minkowski'))
        cv_scores.append(scores_mean)

    Range = list(range(1,10))
    MSE = [ x for x in cv_scores]
    plt.figure(4)

    plt.title("Euclidean distance measurement with normalization")
    plt.plot(Range, MSE)
    plt.xlabel('Number of Neighbors K')
    plt.ylabel('Learning level')
    plt.show()
    print(type(scaled_input))
    print("--------------")
    print(type(input))
