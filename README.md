# Curvature–Orientation MLP (MNIST & EMNIST Letters)

**Code and resources for:**  
*“An MLP Baseline for Handwriting Recognition Using Planar Curvature and Gradient Orientation”* — Azam Nouri

---

## Abstract
We study whether **second-order stroke geometry**—captured by planar curvature and gradient orientation—can support high-quality handwritten character recognition (HCR) without convolutions. A compact, fully connected MLP consumes three channels derived from curvature magnitude \(|\kappa|\), curvature sign \(\mathrm{sign}(\kappa)\), and gradient orientation \(\theta\) on \(28\times28\) images. Channels are assembled per image and flattened to a single feature vector; no data augmentation is used. In a single run with a **stratified 80/20 split** of TFDS train+test, the model achieves strong accuracy on MNIST and EMNIST Letters, indicating that much of the class-discriminative structure resides in **how** and **where** strokes bend and are oriented.

---

## TL;DR
- **Input channels (per image):**  
  \(|\kappa|\) (max-normalized to \([0,1]\)), \(\mathrm{sign}(\kappa)\in\{-1,0,1\}\), and \(\theta\in[0,1]\) via \(\theta'=(\arctan2(g_y,g_x)+\pi)/(2\pi)\)  
- **Feature vector:** \(3\times28\times28 = 2352\) dims (stack then flatten)  
- **Model:** MLP with Dense+BN+ReLU+Dropout blocks (2048 → 1024 → 512 → 256 → Softmax)  
- **Training:** Adam (lr \(=10^{-3}\)), EarlyStopping(patience=8), ReduceLROnPlateau(patience=4)  
- **Metric:** top-1 accuracy (balanced datasets + stratified split)

---

## Results (single run; stratified 80/20 split)
| Model | MNIST | EMNIST Letters |
|---|---:|---:|
| Curvature–Orientation MLP (this work) | ~**97.2%** | ~**89.0%** |

> **Metric choice.** MNIST and EMNIST Letters have (near-)uniform per-class counts by design, and the split here is **stratified**. Therefore, **top-1 accuracy** is an appropriate primary metric. In imbalanced deployments, also report macro-F1 / balanced accuracy.

---

## Reproducibility (matches the notebooks)

### Data & split
- **Datasets:** MNIST; EMNIST Letters (via **TensorFlow Datasets — TFDS**).
- **Loading:** Use TFDS **official** `train` and `test`, then **concatenate** and perform a **stratified 80/20** split (`random_state=42`).  
  - EMNIST Letters labels are remapped from `1..26` to `0..25`.
- **Validation:** Keras `validation_split=0.1` on the training portion.

### Feature extraction (per image)
- **Gradients:** OpenCV `cv2.Sobel` with `ksize=3`  
  - \(g_x = \partial_x I\), \(g_y = \partial_y I\) using `CV_32F`  
  - Second derivatives: \(f_{xx}, f_{yy}, f_{xy}\) via Sobel orders `(2,0)`, `(0,2)`, `(1,1)`
- **Curvature:**  
  \[
  \kappa \;=\; \frac{f_{xx}\,g_y^2 - 2\,f_{xy}\,g_x g_y + f_{yy}\,g_x^2}{\big(g_x^2 + g_y^2 + 10^{-8}\big)^{1.5} + 10^{-8}}
  \]
  - \(|\kappa|\) **max-normalized per image** to \([0,1]\) (divide by its own max if \(>0\))  
  - \(\mathrm{sign}(\kappa)\) kept as \(-1/0/+1\) (no further scaling)
- **Orientation:** \(\theta' = (\arctan2(g_y,g_x) + \pi) / (2\pi)\in[0,1]\)
- **Final features:** stack \(|\kappa|\), \(\mathrm{sign}(\kappa)\), \(\theta'\) → shape \(3\times28\times28\) → **flatten to 2352**  
  *(No global standardization; any `StandardScaler` code is commented out.)*
- **Note:** Raw TFDS images (uint8) are passed directly to Sobel; OpenCV converts to `float32` via `CV_32F`.

### Architecture
- `Input(2352)`  
- `Dense(2048) → BN → ReLU → Dropout(0.5)`  
- `Dense(1024) → BN → ReLU → Dropout(0.5)`  
- `Dense(512)  → BN → ReLU → Dropout(0.4)`  
- `Dense(256)  → BN → ReLU → Dropout(0.3)`  
- `Dense(10)` (MNIST) / `Dense(26)` (Letters) + Softmax

### Optimisation & training
- **Optimizer:** Adam with `learning_rate=1e-3`
- **Loss:** `sparse_categorical_crossentropy`
- **Batch size:** 128
- **Epochs:** up to **100**
- **Callbacks:**  
  - `EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True)`  
  - `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)`
- **Augmentation/regularization:** no data augmentation; no explicit L2 weight decay.

---

## Notebooks
- `notebooks/Curvature_Orientation_pipeline_MNIST.ipynb`  
- `notebooks/Curvature_Orientation_pipeline_EMNIST_Letters.ipynb`

---

## Environment & Installation
```bash
# Python 3.9+ recommended
pip install --upgrade pip

# Core deps
pip install tensorflow tensorflow-datasets opencv-python tqdm scikit-learn numpy

Limitations
Not rotation/shift invariant; modest perturbations can degrade accuracy.

No robustness or latency/energy measurements reported here.

Results limited to MNIST and EMNIST Letters.

License
This repository is licensed under the Apache-2.0 License. See LICENSE.

Contact
Azam Nouri — azamnouri2024@gmail.com

How to Cite
If you use this repository, please cite:
@misc{nouri2025curvmlp,
  title        = {An MLP Baseline for Handwriting Recognition Using Planar Curvature and Gradient Orientation},
  author       = {Nouri, Azam},
  year         = {2025},
  howpublished = {\url{https://github.com/MN-21/Curvature-Orientation-MLP}},
  note         = {Code repository}
}
















# handwriting‑dnn‑features

Edge‑aware, lightweight deep‑learning pipelines for handwritten character recognition. Instead of feeding raw pixels into a CNN, I **explicitly extract shape primitives** that matter:

| Pipeline                      | Feature maps (channels)   | What it captures                         |
|------------------------------|----------------------------|------------------------------------------|
| **Sobel‑Gradient MLP**        | `Gx`, `Gy`                 | First‑order edge magnitude               |
| **Curvature‑Orientation MLP** | `|κ|`, `sign(κ)`, `θ`       | Second‑order bend + stroke direction     |
| **CNN baseline**              | raw pixels                 | Standard convolutional hierarchy         |

Each feature stack is fed to a *five‑layer multilayer perceptron (~60k parameters)* that learns to classify either **MNIST digits** or **EMNIST letters** from scratch—no pre‑training, no transfer learning.

---




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



Apache 2.0 — free for academic & commercial use. See [LICENSE](LICENSE).

---

*Questions or ideas? Open an issue or ping me on GitHub.*
