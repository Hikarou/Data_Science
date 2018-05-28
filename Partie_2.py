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
    breast_cancer = datasets.load_breast_cancer()
    breast_data = preprocessing.normalize(breast_cancer['data'])
    breast_target = breast_cancer['target']

    print(breast_cancer)

    wine = datasets.load_wine()
    wine_data = wine['data']
    wine_target = wine['target']

    kf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None)

    n_neigh_scores = []
    for n_neighbors in range(1, 11):
        local_scores = []
        for train_index, test_index in kf.split(breast_data):
            # print("Train:", train_index, "Validation:", test_index)
            X_train, X_test = breast_data[train_index], breast_data[test_index]
            y_train, y_test = breast_target[train_index], breast_target[test_index]

            k_neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
            k_neigh.fit(X_train, y_train)

            predictions = k_neigh.predict(X_test)
            local_scores.append(accuracy_score(y_test, predictions))
        print(local_scores)
        n_neigh_scores.append(np.mean(np.array(local_scores)))
    print(n_neigh_scores)
    n_neigh_mean = np.mean(np.array(n_neigh_scores))
    print("n_neigh mean : {}".format(n_neigh_mean))

    pass

    tree = DecisionTreeClassifier()
    mlp = MLPClassifier()


if __name__ == '__main__':
    main()
