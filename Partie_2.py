# http://scikit-learn.org/stable/datasets/index.html#breast-cancer-wisconsin-diagnostic-database

from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
import numpy as np


def main() -> None:
    datas = [datasets.load_breast_cancer, datasets.load_wine]
    datas_name = ["Breast Cancer", "Wine"]

    classifiers = []

    # KNeighborsClassifier part
    sub_classifiers = []
    for n_neighbors in range(1, 52, 5):
        sub_classifiers.append(KNeighborsClassifier(n_neighbors=n_neighbors))
    classifiers.append(sub_classifiers)

    # DecisionTreeClassifier
    sub_classifiers = []
    for min_samples_leaf in range(1, 52, 5):
        sub_classifiers.append(DecisionTreeClassifier(min_samples_leaf=min_samples_leaf))
    classifiers.append(sub_classifiers)

    # MLP
    classifiers.append([MLPClassifier(solver='sgd', activation='logistic', max_iter=1000, hidden_layer_sizes=(5, 5),
                                      verbose=False, learning_rate_init=0.1, tol=0., early_stopping=True)])
    classifiers.append([MLPClassifier(solver='adam', activation='logistic', max_iter=1000, hidden_layer_sizes=(5, 5),
                                      verbose=False, learning_rate_init=0.1, tol=0., early_stopping=True)])

    classifiers_name = ["KNeighborsClassifier", "DecisionTreeClassifier", "MLPClassifier_sgd_solver",
                        "MLPClassifier_adam_solver"]

    kf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None)

    for data, data_name in zip(datas, datas_name):
        raw = data()
        print("{} data :".format(data_name))
        x = preprocessing.normalize(raw['data'])
        y = raw['target']

        # k_neigh = KNeighborsClassifier()
        # param_grid = dict(n_neighbors=[1, 5, 15, 25])
        # grid = GridSearchCV(k_neigh, param_grid, cv=5, n_jobs=10, scoring='accuracy')
        #
        # grid.fit(x, y)
        # print(grid.cv_results_['mean_test_score'])

        for sub_classifiers, name in zip(classifiers, classifiers_name):
            scores = []
            for classifier in sub_classifiers:
                local_scores = []
                for train_index, test_index in kf.split(x):
                    x_train, x_test = x[train_index], x[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    classifier.fit(x_train, y_train)

                    predictions = classifier.predict(x_test)
                    local_scores.append(accuracy_score(y_test, predictions))
                scores.append(np.mean(np.array(local_scores)))
            mean = np.mean(np.array(scores))
            print("{} mean is {}".format(name, mean))


if __name__ == '__main__':
    main()
