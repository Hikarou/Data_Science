from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def main() -> None:
    titanic_data = np.loadtxt('titanic.dat', delimiter=',', skiprows=8)

    means = []  # mean value for each K
    K = list(range(1, 25))

    for i in K:  # each K
        KM = KMeans(n_clusters=i, random_state=0).fit(titanic_data)
        
        labels = KM.labels_
        clusters_distances = KM.transform(titanic_data)
        mean_distances = []  # mean distances for each cluster and its points

        for j in range(i):  # index of each cluster
            n_points = 0
            result = 0
            for k in range(len(titanic_data)):  # index of each data point
                if (labels[k] == j):
                    n_points += 1
                    result += clusters_distances[k][j]
            mean_distances.append(result / n_points)

        means.append(np.mean(mean_distances))
    
    plt.plot(K, means)
    plt.title('Valeur moyenne en fonction de K', fontsize=16)
    plt.ylabel('Valeur moyenne')
    plt.xlabel('K')
    plt.show()
    
"""
    x_survived = []
    y_survived = []
    z_survived = []
    x_dead = []
    y_dead = []
    z_dead = []
    for i in titanic_data:
        if (i[3] == 1.0):
            x_survived.append(i[0])
            y_survived.append(i[1])
            z_survived.append(i[2])
        else:
            x_dead.append(i[0])
            y_dead.append(i[1])
            z_dead.append(i[2])
        
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x_survived, y_survived, z_survived, c='r');
    ax.scatter3D(x_dead, y_dead, z_dead, c='b');

    ax.set_title('Donnees', fontsize=16)
    ax.set_xlabel('billet')
    ax.set_ylabel('age')
    ax.set_zlabel('sexe')

    plt.show()"""

if __name__ == '__main__':
    main()
