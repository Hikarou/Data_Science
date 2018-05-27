from sklearn.cluster import KMeans
import numpy as np

titanic_data = np.loadtxt('titanic.dat', delimiter=',', skiprows=8)

means = []  # mean value for each K

for i in range(1, 20):  # each K
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

print(means)
