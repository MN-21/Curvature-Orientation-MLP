# Curvature–Orientation MLP (MNIST & EMNIST Letters)

**Code and resources for:**  
“An MLP Baseline for Handwriting Recognition Using Planar Curvature and Gradient Orientation” — Azam Nouri

---

## Abstract
We test whether second-order stroke geometry — captured by planar curvature and gradient orientation — can support handwritten character recognition without any convolutions. A compact MLP consumes three channels per image: curvature magnitude, curvature sign, and gradient orientation (normalized). With a stratified 80/20 train/test split (from TFDS train+test) and no data augmentation, the model reaches strong accuracy on MNIST and EMNIST Letters.

---

## TL;DR
- **Inputs (3 channels per image):**  
  1) curvature magnitude, 2) curvature sign (−1/0/+1), 3) gradient orientation normalized to [0, 1]  
- **Feature vector:** 3 × 28 × 28 = 2,352 dims (stack then flatten)  
- **Model:** MLP with Dense + BatchNorm + ReLU + Dropout blocks  
- **Training:** Adam, early stopping, reduce-on-plateau  
- **Metric:** top-1 accuracy (datasets are balanced; split is stratified)

---

## Results (single run; stratified 80/20 split)
| Model                          | MNIST   | EMNIST Letters |
|--------------------------------|--------:|---------------:|
| Curvature–Orientation MLP      | **97.2%** | **89.0%**      |

> **Metric choice.** MNIST and EMNIST Letters are (near-)uniform per class, and the split is stratified, so top-1 accuracy is an appropriate primary metric. For imbalanced deployments, also report macro-F1 or balanced accuracy.

---

## Reproducibility (matches the notebooks)

### Data & split
- **Datasets:** MNIST; EMNIST Letters (via TensorFlow Datasets, “TFDS”).  
- **Split:** Concatenate TFDS `train` and `test`, then make a stratified 80/20 split (`random_state=42`).  
- **Validation:** During training, Keras holds out 10% of the training portion via `validation_split=0.1`.  
- **Labels:** EMNIST Letters labels 1–26 are remapped to 0–25.

### Feature extraction (per image)
- **Gradients:** OpenCV `cv2.Sobel`, kernel size 3.  
  - First derivatives: `gx = dI/dx`, `gy = dI/dy`  
  - Second derivatives: `fxx`, `fyy`, `fxy` via Sobel orders (2,0), (0,2), and (1,1)
- **Curvature (kappa):** computed from first and second derivatives with small epsilons in the denominator for stability.  
  - Store **curvature magnitude** as a [0, 1] channel (normalize by its per-image max if > 0).  
  - Store **curvature sign** as a separate channel with values in {−1, 0, +1}.  
- **Orientation:** `theta_norm = (atan2(gy, gx) + pi) / (2*pi)` → values in [0, 1].  
- **Final features:** stack [curvature_mag, curvature_sign, theta_norm] → shape 3×28×28 → **flatten to 2,352**.  
- **Note:** No global standardization; no data augmentation.

### Architecture
- Input(2352)  
- Dense(2048) → BatchNorm → ReLU → Dropout(0.5)  
- Dense(1024) → BatchNorm → ReLU → Dropout(0.5)  
- Dense(512)  → BatchNorm → ReLU → Dropout(0.4)  
- Dense(256)  → BatchNorm → ReLU → Dropout(0.3)  
- Output: Dense(10) for MNIST or Dense(26) for EMNIST Letters → Softmax

### Optimization & training
- Optimizer: **Adam** (learning_rate = 1e-3)  
- Loss: `sparse_categorical_crossentropy`  
- Batch size: 128  
- Epochs: up to 100  
- Callbacks:  
  - `EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True)`  
  - `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)`  
- Regularization: no explicit L2; no augmentation.

---

## Notebooks
- `notebooks/Curvature_Orientation_pipeline_MNIST.ipynb`  
- `notebooks/Curvature_Orientation_pipeline_EMNIST_Letters.ipynb`  
*(Optional for context: `notebooks/CNN_Baseline_EMNIST_Letters.ipynb`)*

---

## Environment & Installation
```bash
# Python 3.9+ recommended
pip install --upgrade pip

# Core dependencies
pip install tensorflow tensorflow-datasets opencv-python tqdm scikit-learn numpy
```
## License
This repository is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Contact
Azam Nouri — azamnouri2024@gmail.com

## How to Cite (Sobel-Gradient MLP)
If you use the **Sobel-Gradient MLP** repository, please cite:
```bibtex
@misc{nouri2025curvmlp,
  title        = {An MLP Baseline for Handwriting Recognition Using Planar Curvature and Gradient Orientation},
  author       = {Nouri, Azam},
  year         = {2025},
  howpublished = {\url{https://github.com/MN-21/Curvature-Orientation-MLP}},
  note         = {Code repository}
}

```

*Questions or ideas? Open an issue or ping me on GitHub.*
