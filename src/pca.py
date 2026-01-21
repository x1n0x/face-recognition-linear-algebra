import numpy as np


def compute_mean_face(X):
    return np.mean(X, axis=1)


def center_data(X, mean_face):
    return X - mean_face[:, np.newaxis]


def eigenfaces_from_small_covariance(X_centered):
    """
    Classical Eigenfaces trick:
    uses small covariance matrix when m << n

    X_centered: shape (n, m)
    returns:
        eigenvalues (m,)
        eigenfaces (n, m)
    """
    m = X_centered.shape[1]

    # small covariance matrix (m x m)
    C_small = (X_centered.T @ X_centered) / m

    # eigen-decomposition
    eigenvalues, eigenvectors_small = np.linalg.eigh(C_small)

    # sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors_small = eigenvectors_small[:, idx]

    # project back to original space
    eigenfaces = X_centered @ eigenvectors_small

    # normalize eigenfaces
    norms = np.linalg.norm(eigenfaces, axis=0)
    eigenfaces = eigenfaces / norms

    return eigenvalues, eigenfaces
