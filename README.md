# Face Recognition via Linear Algebra

This project implements a face recognition system based purely on linear algebra.
Facial images are represented as vectors in a high-dimensional space, and identity
is modeled as a linear subspace learned via PCA (Eigenfaces).

The project intentionally avoids deep learning, gradient descent, and black-box
models. All core ideas are implemented from scratch using NumPy, with a focus on
geometric interpretation, transparency, and limitations of linear methods.

---

## Motivation

This project was developed immediately after completing the course  
**Mathematics for Machine Learning: Linear Algebra**  
by **Imperial College London**.

The main goal was to translate abstract linear algebra concepts —
vector spaces, projections, eigenvalues, eigenvectors, and subspaces —
into a concrete and interpretable machine learning application.

Rather than focusing on performance or modern architectures,
the project emphasizes understanding how mathematical structure alone
can be used to solve a real-world problem such as face recognition.

---

## Core Idea

- Each face image is represented as a vector in ℝⁿ.
- A set of face images forms a point cloud in a high-dimensional space.
- PCA is applied to identify principal directions of variation (eigenfaces).
- These directions define a low-dimensional linear subspace.
- A face is projected into this subspace.
- Recognition is performed by measuring reconstruction error
  (distance to the eigenface subspace).
- A simple threshold determines whether the face is accepted or rejected.

No classifiers, no neural networks, and no optimization procedures are used.

---

## Project Structure

face-recognition-linear-algebra/
│
├── notebooks/
│ ├── 01_faces_as_vectors.ipynb
│ ├── 02_mean_face_and_centering.ipynb
│ ├── 03_eigenfaces_pca_from_scratch.ipynb
│ ├── 04_projection_and_identity_distance.ipynb
│ └── 05_failure_cases_and_limitations.ipynb
│
├── src/
│ ├── data_utils.py
│ ├── pca.py
│ ├── eigenfaces.py
│ ├── recognition.py
│ └── demo.py
│
├── data/
│ └── requirements.txt
│
├── requirements.txt
└── README.md


- `notebooks/` contain a step-by-step mathematical explanation.
- `src/` contains a clean implementation of the eigenfaces pipeline.
- `demo.py` provides a runnable end-to-end example.
- Face images are not included in the repository.

---

## Dataset

The project uses the **AT&T (ORL) Face Database**.

Dataset source:  
https://www.kaggle.com/datasets/kasikrit/att-database-of-faces

Images are grayscale (.pgm) with resolution 92×112.
The dataset is commonly used in classical eigenfaces experiments.

Expected structure:

data/
├── train_faces/
│ ├── s1/
│ ├── s2/
│ └── ...
└── test_faces/
└── sX/
└── image.pgm

---

## Running the Demo

From the project root:

```bash
python src/demo.py
```

---
## The Demo

The demo script performs the following steps:

- Builds an eigenfaces model from training images.
- Projects a test face into the eigenface subspace.
- Computes the reconstruction error.
- Applies a threshold-based accept / reject decision.
- Visualizes the result and provides a human-readable explanation.

---

## Acceptance vs Rejection Example

The following example illustrates how the model makes a decision.

The eigenface subspace is learned from images of one subject.
A face from a different subject is then evaluated.

<img width="1000" height="400" alt="Figure_1" src="https://github.com/user-attachments/assets/fc2c4f9b-1f2e-4f59-9639-838529b9cd21" />

**Decision:** REJECTED  
**Distance:** (insert value)

**Explanation:**
- The test face cannot be accurately reconstructed from the eigenface
  subspace learned on the training identity.
- The reconstruction error is large.
- The face is therefore rejected as belonging to a different identity.

---

## Interpretation

The model does not compare faces directly.

Instead, it answers the following geometric question:

> Can this face be well explained by the learned identity subspace?

- Faces close to the subspace are well reconstructed and accepted.
- Faces far from the subspace produce large reconstruction error and are rejected.

This makes the decision process explicit and interpretable.

---

## Limitations

- Sensitivity to illumination and pose.
- Linear approximation of a nonlinear problem.
- Identity modeled only in pixel space.
- Threshold selection is dataset-dependent.

These limitations are discussed explicitly in the final notebook.

---

## Conclusion

This project demonstrates that face recognition can be formulated
as a geometric problem using linear algebra alone.

Eigenfaces provide an interpretable baseline that connects abstract
linear algebra concepts to a practical machine learning task,
while clearly exposing the strengths and limitations of linear models.

