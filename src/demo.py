"""
Eigenfaces demo with reconstruction visualization.
Run with:
    python src/demo.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# -------------------------------------------------
# PATH FIX (script mode)
# -------------------------------------------------

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.append(str(SRC_DIR))

DATA_DIR = PROJECT_ROOT / "data"

# -------------------------------------------------
# IMPORTS
# -------------------------------------------------

from data_utils import load_face_image, image_to_vector
from pca import compute_mean_face, center_data, eigenfaces_from_small_covariance
from eigenfaces import project_face, reconstruct_face
from recognition import recognize_face

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

IMAGE_WIDTH = 92
IMAGE_HEIGHT = 112

IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)      # для PIL
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH)     # для NumPy

NUM_COMPONENTS = 10
THRESHOLD = 3000.0

TRAIN_DIR = DATA_DIR / "train_faces" / "s1"
TEST_IMAGE = DATA_DIR / "train_faces" / "s1" / "1.pgm"

# -------------------------------------------------
# VISUALIZATION
# -------------------------------------------------

def show_reconstruction(original, reconstructed, image_shape):
    original_img = original.reshape(image_shape, order="C")
    reconstructed_img = reconstructed.reshape(image_shape, order="C")
    diff_img = original_img - reconstructed_img

    plt.figure(figsize=(9, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(original_img, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(reconstructed_img, cmap="gray")
    plt.title("Reconstruction")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(diff_img, cmap="gray")
    plt.title("Difference")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# -------------------------------------------------
# LOAD TRAINING DATA
# -------------------------------------------------

print("Loading training images from:", TRAIN_DIR)

image_paths = list(TRAIN_DIR.rglob("*.pgm"))
if len(image_paths) == 0:
    raise RuntimeError("No training images found.")

vectors = []
for p in image_paths:
    img = load_face_image(p, size=IMAGE_SIZE)
    vectors.append(image_to_vector(img))

X = np.column_stack(vectors)
print("Training matrix shape:", X.shape)

# -------------------------------------------------
# PCA (EIGENFACES)
# -------------------------------------------------

print("Computing mean face...")
mean_face = compute_mean_face(X)

print("Centering data...")
X_centered = center_data(X, mean_face)

print("Computing eigenfaces (small covariance trick)...")
eigenvalues, eigenfaces = eigenfaces_from_small_covariance(X_centered)

k = min(NUM_COMPONENTS, eigenfaces.shape[1])
U_k = eigenfaces[:, :k]

# -------------------------------------------------
# LOAD TEST IMAGE
# -------------------------------------------------

print("Testing image:", TEST_IMAGE)

img = load_face_image(TEST_IMAGE, size=IMAGE_SIZE)
x = image_to_vector(img)
x_centered = x - mean_face

# -------------------------------------------------
# RECOGNITION
# -------------------------------------------------

accepted, distance = recognize_face(
    x_centered=x_centered,
    U_k=U_k,
    threshold=THRESHOLD
)

print("\n--- RESULT ---")
print(f"Distance: {distance:.2f}")
print("Decision:", "ACCEPTED" if accepted else "REJECTED")

# -------------------------------------------------
# RECONSTRUCTION + VISUALIZATION
# -------------------------------------------------

y = project_face(x_centered, U_k)
x_hat = reconstruct_face(y, U_k)

show_reconstruction(
    original=x,
    reconstructed=x_hat + mean_face,
    image_shape=IMAGE_SHAPE
)
