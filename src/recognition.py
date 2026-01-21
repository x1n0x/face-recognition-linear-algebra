from eigenfaces import distance_to_subspace


def recognize_face(x_centered, U_k, threshold):
    distance = distance_to_subspace(x_centered, U_k)
    return distance <= threshold, distance
