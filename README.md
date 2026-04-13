# Credit Card Fraud Detection using Ensemble Learning

This project implements a robust machine learning pipeline to detect fraudulent credit card transactions using an ensemble of Logistic Regression, Random Forest, and Multi-layer Perceptron (MLP) Neural Network.

## Project Architecture
* **Modular Codebase:** Organized into training, evaluation, and data utility scripts.
* **Feature Engineering:** Implemented **Cyclic Encoding (Sine/Cosine)** for time-based features to preserve temporal relationships.
* **Handling Imbalance:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the dataset.
* **Ensemble Strategy:** Combined multiple classifiers to improve stability and recall.

## Final Results (On Test Set)
After training on the combined training and validation sets:
* **F1-Score:** 0.8241
* **Confusion Matrix:**
  * True Negatives: 56,843
  * False Positives: 20
  * False Negatives: 15
  * True Positives: 82

*Note: The original dataset is private and not included in this repository.*
