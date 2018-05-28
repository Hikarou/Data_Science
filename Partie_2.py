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

    kf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None)

    for data in datas:
        raw = data()
        print(raw['DESCR'].split(' ', 1)[0], "data :")
        X = preprocessing.normalize(raw['data'])
        y = raw['target']

        # k_neigh = KNeighborsClassifier()
        # param_grid = dict(n_neighbors=[1, 5, 15, 25])
        # grid = GridSearchCV(k_neigh, param_grid, cv=5, n_jobs=10, scoring='accuracy')
        #
        # grid.fit(X, y)
        # print(grid.cv_results_['mean_test_score'])

        # KNeighborsClassifier part
        n_neigh_scores = []
        for n_neighbors in range(1, 52, 5):
            local_scores = []
            for train_index, test_index in kf.split(X):
                # print("Train:", train_index, "Validation:", test_index)
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                k_neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
                k_neigh.fit(X_train, y_train)

                predictions = k_neigh.predict(X_test)
                local_scores.append(accuracy_score(y_test, predictions))
            # print(local_scores)
            n_neigh_scores.append(np.mean(np.array(local_scores)))
        # print(n_neigh_scores)
        n_neigh_mean = np.mean(np.array(n_neigh_scores))
        print("n_neigh mean : {}".format(n_neigh_mean))

        # DecisionTreeClassifier
        tree_scores = []
        for min_samples_leaf in range(1, 52, 5):
            local_scores = []
            for train_index, test_index in kf.split(X):
                # print("Train:", train_index, "Validation:", test_index)
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
                model.fit(X_train, y_train)

                predictions = model.predict(X_test)
                local_scores.append(accuracy_score(y_test, predictions))
                # print(local_scores)
                tree_scores.append(np.mean(np.array(local_scores)))
        # print(tree_scores)
        tree_mean = np.mean(np.array(tree_scores))
        print("tree mean : {}".format(tree_mean))

    pass

    mlp = MLPClassifier()


if __name__ == '__main__':
    main()
