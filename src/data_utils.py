import numpy as np
from PIL import Image


def load_face_image(path, size=(112, 92)):
    image = Image.open(path).convert("L")
    image = image.resize(size)
    return np.asarray(image, dtype=np.float64)


def image_to_vector(image):
    return image.reshape(-1, order="C")

