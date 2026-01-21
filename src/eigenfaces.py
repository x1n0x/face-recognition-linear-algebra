import numpy as np


def project_face(x_centered, U_k):
    return U_k.T @ x_centered


def reconstruct_face(y, U_k):
    return U_k @ y


def distance_to_subspace(x_centered, U_k):
    y = project_face(x_centered, U_k)
    x_hat = reconstruct_face(y, U_k)
    return np.linalg.norm(x_centered - x_hat)
