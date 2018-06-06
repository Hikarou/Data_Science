from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def main() -> None:
    titanic_data = np.loadtxt('titanic.dat', delimiter=',', skiprows=8)    
    
    """Mean value"""
    means = []  # mean value for each K
    K = list(range(1, 25))
    
    for i in K:  # each K
        KM = KMeans(n_clusters=i, init='random', random_state=0).fit(titanic_data)
        
        labels = KM.labels_
        clusters_distances = KM.transform(titanic_data)
        mean_distances = []  # mean distances for each cluster and its points

        for j in range(i):  # index of each cluster
            n_points = 0
            result = 0
            for k in range(len(titanic_data)):  # index of each data point
                if labels[k] == j:
                    n_points += 1
                    result += clusters_distances[k][j]
            if n_points == 0: #if no point is close to this cluster (only at initialization)
                continue;
            else:        
                mean_distances.append(result / n_points)

        means.append(np.mean(mean_distances))
    
    plt.plot(K, means)
    plt.title('Valeur moyenne en fonction de K', fontsize=16)
    plt.ylabel('Valeur moyenne')
    plt.xlabel('K')
    plt.show()  
    
    """Data display"""
    KM = KMeans(n_clusters=8, init='random', random_state=0).fit(titanic_data) #8: chosen value for K
    centers = KM.cluster_centers_
    x_centers = []
    y_centers = []
    z_centers = []
    for i in centers: #3 vectors each with one of the coordinates of the centers points
        x_centers.append(i[0])
        y_centers.append(i[1])
        z_centers.append(i[2])
        
    x_dead = []
    y_dead = []
    z_dead = []    
    x_survived = []
    y_survived = []
    z_survived = []
    for i in titanic_data:
        if i[3] == -1.0: #3 vectors each with one of the coordinates of the "dead" points
            x_dead.append(i[0])
            y_dead.append(i[1])
            z_dead.append(i[2])
        else: #3 vectors each with one of the coordinates of the "survived" points
            x_survived.append(i[0])
            y_survived.append(i[1])
            z_survived.append(i[2])
        
    """3d"""
    ax: Axes3D = plt.axes(projection='3d')
    
    ax.scatter3D(x_dead, y_dead, z_dead, s=50, c='r', marker='s', label='morts');
    ax.scatter3D(x_survived, y_survived, z_survived, s=25, c='y', marker='o', label='rescapes');
    ax.scatter3D(x_centers, y_centers, z_centers, s=30, c='black', marker='x', label='barycentres');

    ax.set_title('Donnees', fontsize=16)
    ax.set_xlabel('billet:[1ere, 2eme, 3eme, equipage]')
    ax.set_ylabel('age:[adultes, enfants]')
    ax.set_zlabel('sexe:[femmes, hommes]')

    plt.legend(loc='best');
    plt.show()
    
    """2d:age/billet"""
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(x_dead, y_dead, s=50, c='r', marker='s', label='morts');
    ax1.scatter(x_survived, y_survived, s=25, c='y', marker='o', label='rescapes');
    ax1.scatter(x_centers, y_centers, s=30, c='black', marker="x", label='barycentres')
    
    ax1.set_title('Donnees: age/billet', fontsize=16)
    ax1.set_xlabel('billet:[1ere, 2eme, 3eme, equipage]')
    ax1.set_ylabel('age:[adultes, enfants]')
    
    plt.legend(loc='best');
    plt.show()
    
    """2d:sexe/age"""
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(y_dead, z_dead, s=50, c='r', marker='s', label='morts');
    ax1.scatter(y_survived, z_survived, s=25, c='y', marker='o', label='rescapes');
    ax1.scatter(y_centers, z_centers, s=30, c='black', marker="x", label='barycentres')
    
    ax1.set_title('Donnees: sexe/age', fontsize=16)
    ax1.set_xlabel('age:[adultes, enfants]')
    ax1.set_ylabel('sexe:[femmes, hommes]')
    
    plt.legend(loc='best');
    plt.show()
    
    """2d:sexe/billet"""
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(x_dead, z_dead, s=50, c='r', marker='s', label='morts');
    ax1.scatter(x_survived, z_survived, s=25, c='y', marker='o', label='rescapes');
    ax1.scatter(x_centers, z_centers, s=30, c='black', marker="x", label='barycentres')
    
    ax1.set_title('Donnees: sexe/billet', fontsize=16)
    ax1.set_xlabel('billet:[1ere, 2eme, 3eme, equipage]')
    ax1.set_ylabel('sexe:[femmes, hommes]')
    
    plt.legend(loc='right');
    plt.show()

if __name__ == '__main__':
    main()
