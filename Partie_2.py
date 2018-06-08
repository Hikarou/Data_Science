# http://scikit-learn.org/stable/datasets/index.html#breast-cancer-wisconsin-diagnostic-database

from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
import numpy as np
import time


def main() -> None:
    datas = [datasets.load_breast_cancer, datasets.load_wine]
    datas_name = ["Breast Cancer", "Wine"]

    classifiers = [KNeighborsClassifier, DecisionTreeClassifier, MLPClassifier, MLPClassifier, MLPClassifier,
                   MLPClassifier]

    params = []

    # KNeighborsClassifier
    sub_params = []
    for n_neighbors in range(1, 52, 5):
        sub_params.append({"n_neighbors": n_neighbors, 'n_jobs': -1})
    params.append(sub_params)

    # DecisionTreeClassifier
    sub_params = []
    for min_samples_leaf in range(2, 53, 5):
        sub_params.append({"min_samples_leaf": min_samples_leaf})
    params.append(sub_params)

    # MLPClassifier one layer sgd_solver
    sub_params = []
    for nodes in range(2, 20, 3):
        sub_params.append({"solver": 'sgd', "activation": 'logistic', "max_iter": 1000, "hidden_layer_sizes": (nodes,),
                           "verbose": False, "learning_rate_init": 0.1, "tol": 0., "early_stopping": True})
    params.append(sub_params)

    # MLPClassifier one layer adam_solver
    sub_params = []
    for nodes in range(2, 20, 3):
        sub_params.append({"solver": 'adam', "activation": 'logistic', "max_iter": 1000, "hidden_layer_sizes": (nodes,),
                           "verbose": False, "learning_rate_init": 0.1, "tol": 0., "early_stopping": True})
    params.append(sub_params)

    # MLPClassifier two layers sgd_solver
    sub_params = []
    for nodes in range(2, 20, 3):
        sub_params.append(
            {"solver": 'sgd', "activation": 'logistic', "max_iter": 1000, "hidden_layer_sizes": (nodes, 5),
             "verbose": False, "learning_rate_init": 0.1, "tol": 0., "early_stopping": True})
    params.append(sub_params)

    # MLPClassifier two layers adam_solver
    sub_params = []
    for nodes in range(2, 20, 3):
        sub_params.append(
            {"solver": 'adam', "activation": 'logistic', "max_iter": 1000, "hidden_layer_sizes": (nodes, 5),
             "verbose": False, "learning_rate_init": 0.1, "tol": 0., "early_stopping": True})
    params.append(sub_params)

    classifiers_name = ["KNeighborsClassifier", "DecisionTreeClassifier", "MLPClassifier one layer sgd_solver",
                        "MLPClassifier one layer adam_solver",
                        "MLPClassifier two layers, second layer 5 nodes, sgd_solver",
                        "MLPClassifier two layers, second layer 5 nodes, adam_solver"]

    kf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None)

    for data, data_name in zip(datas, datas_name):
        raw = data()
        print("{} data of size {} :".format(data_name, raw['data'].shape[0]))
        x = preprocessing.normalize(raw['data'])
        y = raw['target']

        for classifier, name, sub_param in zip(classifiers, classifiers_name, params):
            scores = []
            start = time.perf_counter()
            for param in sub_param:
                local_scores = []
                for train_index, test_index in kf.split(x):
                    x_train, x_test = x[train_index], x[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    model = classifier(**param)
                    model.fit(x_train, y_train)

                    predictions = model.predict(x_test)
                    local_scores.append(accuracy_score(y_test, predictions))
                scores.append(np.mean(np.array(local_scores)))
            end = time.perf_counter()
            np_arr = np.array(scores)
            mean = np.mean(np_arr)
            standard_deviation = np.std(np_arr)
            print("{}\n\tmean is {} with {} as standard deviation.\n\tWorked for {} seconds".format(name, mean,
                                                                                              standard_deviation,
                                                                                              end - start))


if __name__ == '__main__':
    main()
