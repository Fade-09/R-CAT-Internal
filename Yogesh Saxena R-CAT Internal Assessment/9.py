import numpy as np
def pca(X, num_components):
    X_centered = X - np.mean(X, axis=0)
    covariance_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_index = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_index]
    sorted_eigenvectors = eigenvectors[:, sorted_index]
    eigenvector_subset = sorted_eigenvectors[:, :num_components]
    X_reduced = np.dot(X_centered, eigenvector_subset)
    
    return X_reduced, sorted_eigenvalues, sorted_eigenvectors

X = np.array([[2.5, 2.4],
              [0.5, 0.7],
              [2.2, 2.9],
              [1.9, 2.2],
              [3.1, 3.0],
              [2.3, 2.7],
              [2, 1.6],
              [1, 1.1],
              [1.5, 1.6],
              [1.1, 0.9]])

num_components = 1
X_reduced, eigenvalues, eigenvectors = pca(X, num_components)

print("Reduced Data:\n", X_reduced)
print("Eigenvalues:\n", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
