# handwriting-dnn-features
handwriting-dnn-features

This repository contains the full implementation of feature-extraction pipelines and deep neural network models for handwritten character recognition using classical Sobel gradients and curvature–orientation maps as inputs.

Repository Structure

notebooks/

sobel_pipeline.ipynb: Colab notebook demonstrating the Sobel-gradient MLP on EMNIST MNIST and EMNIST Letters.

curvature_orientation_pipeline.ipynb: Colab notebook demonstrating the curvature–orientation MLP on both datasets.

scripts/

feature_extraction.py: Python script containing compute_gradients and extract_curv_and_orientation functions.

train_mlp.py: Training script for both pipelines with command-line arguments.

requirements.txt: List of Python dependencies for reproducibility.

README.md: This documentation file.

Installation

Clone the repository and install dependencies:

git clone https://github.com/MN-21/handwriting-dnn-features.git
cd handwriting-dnn-features
pip install -r requirements.txt

Reproducibility

Random Seeds: All experiments use a fixed seed (42) for Python’s random, NumPy, and TensorFlow for deterministic results.

Dependencies: See requirements.txt for exact package versions.

Results Summary

Pipeline

Dataset

Test Accuracy

Sobel-Gradient MLP

EMNIST MNIST (digits)

98.57%

Sobel-Gradient MLP

EMNIST Letters

92.67%

Curvature–Orientation MLP

EMNIST MNIST (digits)

97.00%

Curvature–Orientation MLP

EMNIST Letters

90.00%

Code Availability

The full implementation is available at: https://github.com/MN-21/handwriting-dnn-features

License

This project is released under the MIT License. See LICENSE for details.

