# handwriting‑dnn‑features

Edge‑aware, lightweight deep‑learning pipelines for handwritten character recognition. Instead of feeding raw pixels into a CNN, I **explicitly extract shape primitives** that matter:

| Pipeline                      | Feature maps (channels)   | What it captures                         |
|------------------------------|----------------------------|------------------------------------------|
| **Sobel‑Gradient MLP**        | `Gx`, `Gy`                 | First‑order edge magnitude               |
| **Curvature‑Orientation MLP** | `|κ|`, `sign(κ)`, `θ`       | Second‑order bend + stroke direction     |
| **CNN baseline**              | raw pixels                 | Standard convolutional hierarchy         |

Each feature stack is fed to a *five‑layer multilayer perceptron (~60k parameters)* that learns to classify either **MNIST digits** or **EMNIST letters** from scratch—no pre‑training, no transfer learning.

---

## What’s inside

```text
handwriting-dnn-features/
├── Sobel_Gradient_pipeline_MNIST.ipynb
├── Sobel_Gradient_pipeline_EMNIST_Letters.ipynb
├── Curvature_Orientation_pipeline_MNIST.ipynb
├── Curvature_Orientation_pipeline_EMNIST_Letters.ipynb
├── CNN_Baseline_EMNIST_Letters.ipynb
└── LICENSE
```

> **Why no `requirements.txt`?** Colab handles the environment automatically, and each notebook installs any missing packages on the fly. For local runs, see the **Local install** section below.

---

## Quick start (Google Colab)

| Dataset            | Pipeline     | Colab Link |
|--------------------|--------------|------------|
| **MNIST digits**   | Sobel        | [Sobel MNIST](https://colab.research.google.com/github/MN-21/handwriting-dnn-features/blob/main/Sobel_Gradient_pipeline_MNIST.ipynb) |
|                    | Curvature    | [Curvature MNIST](https://colab.research.google.com/github/MN-21/handwriting-dnn-features/blob/main/Curvature_Orientation_pipeline_MNIST.ipynb) |
| **EMNIST letters** | Sobel        | [Sobel EMNIST](https://colab.research.google.com/github/MN-21/handwriting-dnn-features/blob/main/Sobel_Gradient_pipeline_EMNIST_Letters.ipynb) |
|                    | Curvature    | [Curvature EMNIST](https://colab.research.google.com/github/MN-21/handwriting-dnn-features/blob/main/Curvature_Orientation_pipeline_EMNIST_Letters.ipynb) |
|                    | CNN baseline | [CNN EMNIST](https://colab.research.google.com/github/MN-21/handwriting-dnn-features/blob/main/CNN_Baseline_EMNIST_Letters.ipynb) |

Open a notebook, go to **Runtime → Run all**. Each notebook:

1. Downloads the dataset with `tensorflow-datasets`.
2. Computes the requested feature stack.
3. Trains the MLP (or CNN) and prints accuracy & a confusion matrix.

*Typical results on an NVIDIA T4 (Colab):*

| Dataset            | Sobel MLP | Curvature MLP | CNN baseline |
|--------------------|-----------|----------------|---------------|
| **MNIST**          | ≈ 97 %    | **≈ 97 %**     | 99 %          |
| **EMNIST letters** | ≈ 92 %    | **≈ 89 %**     | 94 %          |

---

## Local install

```bash
git clone https://github.com/MN-21/handwriting-dnn-features.git
cd handwriting-dnn-features
python -m venv .venv && source .venv/bin/activate   # optional
pip install tensorflow tensorflow-datasets numpy opencv-python tqdm scikit-learn matplotlib
jupyter lab
```

Then open any notebook from the *What’s inside* section and run it.

---

## Citation

```text
@misc{nouri2025edgeaware,
  author       = {Azam Nouri},
  title        = {An MLP Baseline Using Sobel Gradients for Handwritten Character Recognition},
  year         = {2025},
  howpublished = {GitHub},
  url          = {https://github.com/MN-21/handwriting-dnn-features}
}
```
## Citation

```text
@misc{nouri2025edgeaware,
  author       = {Azam Nouri},
  title        = {Handwriting Recognition with Planar Curvature and Gradient Orientation:A Convolution-Free MLP Baseline},
  year         = {2025},
  howpublished = {GitHub},
  url          = {https://github.com/MN-21/handwriting-dnn-features}
}
```
---

## License

Apache 2.0 — free for academic & commercial use. See [LICENSE](LICENSE).

---

*Questions or ideas? Open an issue or ping me on GitHub.*
