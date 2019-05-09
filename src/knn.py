import os
from sklearn.neighbors import KNeighborsClassifier
from cross_validation import cross_validation
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))

def knn(input):    # input = np.random.permutation(input) #losowe wymieszanie wierszy

    cv_scores = []
    for i in range(1,10):
        scores_mean = cross_validation(input, KNeighborsClassifier(n_neighbors=i))
        cv_scores.append(scores_mean)

    Range = list(range(1,10))
    MSE = [ x for x in cv_scores]
    plt.plot(Range, MSE)
    plt.xlabel('Number of Neighbors K')
    plt.ylabel('Learning level')
    plt.show()
