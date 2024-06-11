import numpy as np
from scipy.spatial.distance import cdist

def kmeans(x, k, no_of_iterations):
    idx = np.random.choice(len(x), k, replace=False)
    centroids = x[idx, :]

    distances = cdist(x, centroids, 'euclidean')
    points = np.array([np.argmin(i) for i in distances])


    for _ in range(no_of_iterations):
        centroids = []
        for idx in range(k):
            temp_cent = x[points == idx].mean(axis=0)
            centroids.append(temp_cent)
        centroids = np.vstack(centroids)
        distances = cdist(x, centroids, 'euclidean')
        points = np.array([np.argmin(i) for i in distances])

    return points

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X_train = StandardScaler().fit_transform(digits.data)

k = 3
no_of_iterations = 100
cluster_labels = kmeans(X_train, k, no_of_iterations)

import matplotlib.pyplot as plt
plt.scatter(X_train[:, 0], X_train[:, 1], c=cluster_labels)
plt.show()