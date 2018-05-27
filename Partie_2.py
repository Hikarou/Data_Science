from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


def main() -> None:
    breast_cancer = datasets.load_breast_cancer()
    wine = datasets.load_wine()
    k_neigh = KNeighborsClassifier(n_neighbors=5)
    k_neigh.fit(breast_cancer, wine)
    tree = DecisionTreeClassifier()
    tree.fit(breast_cancer, wine)
    mlp = MLPClassifier()
    mlp.fit(breast_cancer, wine)


if __name__ == '__main__':
    main()
