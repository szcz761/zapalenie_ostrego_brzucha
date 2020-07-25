import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
from mlxtend.feature_selection import SequentialFeatureSelector
import random

# # a=43 
# a=30
# # count =164
# count =475

def cross_validation(input, classifiers):
    result_array = np.empty((0, 10))
    # print("Cross Validation")
    X, y = input[:, :-1], input[:, -1]
    # X -  dane na podstawie ktorych klayfikujemy - macierz 475x31
    # y -  klasy do ktorych klasyfikujemy (target) - kolumna o indeksie 31
    # print(np.unique(y, return_counts=True))

    # na ile podzbiorw dzielimy 10% to test gdy jest n_split = 10, KFold Stratyfikowany
    # czyli dzielimy podzbiory jednak w sposob inny niz dla problemu binarnego(more accuracy)
    kf = StratifiedKFold(n_splits=10, shuffle=True) # ZWRACA WEKTORY A NIE MACIERZE!!!!!!!
    i=1
    result_array_single_classifier = []
    for train, test in kf.split(X, y):

        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        result_array = []
        
        for classifier in classifiers:
            clf = classifier.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            # gdy mamy wiecej klas niz 2 i nie jest zbalansowane np klasa a ma 123o obiekty a klasa b ma 12 obiektw
            bac = balanced_accuracy_score(y_test, y_pred)

            single_result = clf.score(X_test, y_test)

            # print("normalizacja: ", type(input[0][0]),
            #       "ilosc cech: ", np.shape(X)[1],
            #       "klasyfikator: ", classifier,
            #       "metric: ", classifier.metric,
            #       "SR %.3f | %.3f | bac %.3f" % (
            #        single_result, accuracy, bac
            # ))
            result_array_single_classifier = np.append(result_array_single_classifier, single_result)
            if(i%10) == 0:
                # print("New data:==================================================================")
                # print(np.mean(result_array_single_classifier,dtype=np.float64))
                # print("New data:==================================================================")
                result_array =  np.append(result_array, np.mean(result_array_single_classifier,dtype=np.float64))
                result_array_single_classifier = []
            i=i+1

    return result_array

def create_lower_dimention_matrix_from_filters(input, how_many_attrs, cv_scores):
    target = np.array(input[:,-1]).reshape(count,1)
    for attrs_iter in range(-1,how_many_attrs):
        data_fill = adding_attribute(attrs_iter, input, cv_scores)
    return np.hstack((data_fill, target))

def random_test(input,how_many_attrs, cv_scores):
    random_index = []
    y = np.array(input[:, -1])
    for i in range(0, how_many_attrs):
        random_index.append(random.randint(0, a))

    target = np.array(input[:,-1]).reshape(count,1)
    return np.hstack((input[:,random_index],target))

def kolmogorov_test(input,how_many_attrs, cv_scores):
    if(cv_scores != []):
        return create_lower_dimention_matrix_from_filters(input, how_many_attrs, cv_scores)

    y = np.array(input[:, -1])
    for i in range(0, a):
        one_attribute_data = input[:, i]
        scores_mean = (stats.ks_2samp(y, one_attribute_data), i)
        cv_scores.append(scores_mean)
    cv_scores = sorted(cv_scores, key=lambda x: x[0], reverse=False) #true or fale ??

    return create_lower_dimention_matrix_from_filters(input, how_many_attrs, cv_scores)


def wraper_test(input, how_many_attrs, cv_scores):
    if(cv_scores != []):
        return create_lower_dimention_matrix_from_filters(input, how_many_attrs, cv_scores)

    y = np.array(input[:, -1])
    for i in range(0, a):
        one_attribute_data = input[:, [i,-1]]
        # one_attribute_data_with_y = np.hstack((one_attribute_data, y))
        scores_mean = (cross_validation(one_attribute_data, [KNeighborsClassifier(n_neighbors=5, metric="euclidean")]),i)
        cv_scores.append(scores_mean)
    cv_scores = sorted(cv_scores, key=lambda x: x[0], reverse=False) #true or fale ??

    return create_lower_dimention_matrix_from_filters(input, how_many_attrs, cv_scores)

def pearson_test(input,how_many_attrs, cv_scores):
    if(cv_scores != []):
        return create_lower_dimention_matrix_from_filters(input, how_many_attrs, cv_scores)
    y = np.array(input[:, -1])
    for i in range(0, a):
        one_attribute_data = input[:, i]
        scores_mean = (np.abs(np.corrcoef(y, one_attribute_data)[0,1]), i)
        cv_scores.append(scores_mean)
    cv_scores = sorted(cv_scores, key=lambda x: x[0], reverse=True)

    return create_lower_dimention_matrix_from_filters(input, how_many_attrs, cv_scores)

def PCA_test(input,how_many_attrs, cv_scores):
    x = np.array(input[:, :-1])
    target = np.array(input[:,-1]).reshape(count,1)
    pca = PCA(n_components=how_many_attrs)
    principalComponents = pca.fit_transform(x)
    # principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
    # print(principalComponents)

    return np.hstack((principalComponents, target))

def SFS_test(input,how_many_attrs, cv_scores):
    y = np.array(input[:, -1])
    x = np.array(input[:, :-1])
    # print(how_many_attrs)
    sfs = SequentialFeatureSelector(KNeighborsClassifier(n_neighbors=5, metric="euclidean"), 
               k_features=how_many_attrs, 
               forward=True, 
               floating=False, 
               verbose=0,
               scoring='accuracy',
               n_jobs=-1,
               cv=4)
    sfs = sfs.fit(x, y)
    # print(sfs.k_feature_idx_)
    target = np.array(input[:,-1]).reshape(count,1)
    return np.hstack((input[:,sfs.k_feature_idx_],target))

def input_normalization(input, a_, count_):
    global a
    global count
    a=a_
    count=count_
    data = input[:, :-1]  # skalowane sa tylko dane, nie target
    # target przeksztalcany jest na macierz jednokolumnowa
    target = np.array(input[:, -1]).reshape(count, 1)
    # target = np.array(input[:, -1]).reshape(, 1)
    no_norm = input
    z_score = preprocessing.StandardScaler(
        with_mean=True, with_std=True).fit_transform(data)  # skalowanie z-score
    # skalowanie max abs, czyli przedzialy modulu maksymalnych wartosci
    z_score_out = np.hstack((z_score, target))
    # print(target.shape)
    # print("----------")
    # print(min_max.shape)
    return no_norm, z_score_out


def adding_attribute(how_many_attrs, item, scores):
    attribute_index = 0
    column = np.array(item[:, attribute_index]).reshape(count, 1)
    data_fill = np.array(column)
    for j in range(0, how_many_attrs):
        # iterujemy po cechach o indexie zawartym w tablicy posortowanych cech
        # print(j)
        attribute_index = np.int(scores[j][1])
        column = np.array(item[:, attribute_index]).reshape(count,
                                                            1)  # reshape do macierzy jednokolumnowej, zeby mozna bylo stworzyc macierz cech
        # kolejne kolumny o indeksach z posortowanej listy cech dodajemy do macierzy
        # dodajemy kolumne do macierzy cech
        data_fill = np.hstack((data_fill, column))
    return data_fill
