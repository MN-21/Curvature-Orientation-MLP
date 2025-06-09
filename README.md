# handwriting-dnn-features


Lightweight, edge-aware deep‐learning pipelines for handwritten character recognition.  
Instead of feeding raw pixels into a convolutional neural network (CNN), we **explicitly compute**:

| Pipeline | Channels | Geometry captured |
|----------|----------|-------------------|
| **Sobel-Gradient MLP** | `Gx`, `Gy` | First-order edge strength |
| **Curvature–Orientation MLP** | `|k|`, `sign(k)`, `θ` | Second-order bend intensity, concavity / convexity, stroke direction |

Both pipelines are trained from scratch on **EMNIST** splits with a five-layer multilayer perceptron (MLP).  
The project accompanies the paper:

> Azam Nouri, “Edge-Aware Deep Neural Networks for Hand-Written Character Recognition via Explicit Sobel and Curvature–Orientation Encodings,” 2025.

---

```text
handwriting-dnn-features/
├── Sobel_Gradient_pipeline.ipynb          # notebook: Sobel-based MLP
├── Curvature_Orientation_pipeline.ipynb   # notebook: curvature + orientation MLP
├── CNN_EMNIST.ipynb                       # (optional) CNN baseline notebook
├── requirements.txt                       # Python dependencies
└── LICENSE                                # Apache-2.0 license




*The CNN notebook is kept only as an optional reference; the paper’s experiments do **not** rely on CNN results.*

---

## Quick start (Colab)

| Notebook | Colab link |
|----------|------------|
| Sobel pipeline | <https://colab.research.google.com/github/MN-21/handwriting-dnn-features/blob/main/Sobel_Gradient_pipeline.ipynb> |
| Curvature pipeline | <https://colab.research.google.com/github/MN-21/handwriting-dnn-features/blob/main/Curvature_Orientation_pipeline.ipynb> |

Open in Colab → `Runtime > Run all`.  
Both notebooks download EMNIST via `tensorflow-datasets`, compute features, train, and evaluate.

---

## Local installation

```bash
git clone https://github.com/MN-21/handwriting-dnn-features.git
cd handwriting-dnn-features
python -m venv venv && source venv/bin/activate  # optional
pip install -r requirements.txt
jupyter notebook        # or jupyter lab


