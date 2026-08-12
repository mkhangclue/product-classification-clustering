# Product Classification & Clustering

An unsupervised machine learning project that explores product segmentation using K-Means, Autoencoder-based representation learning, and DBSCAN.

## Project Overview

This project analyzes product data using unsupervised machine learning techniques.

The workflow includes:

- Exploratory Data Analysis
- Feature Engineering
- One-Hot Encoding
- Feature Standardization
- Principal Component Analysis (PCA)
- K-Means clustering baseline
- Autoencoder-based representation learning
- K-Means clustering on the learned latent representation
- DBSCAN clustering
- Cluster analysis and model comparison

The main objective is to evaluate whether a learned latent representation can improve clustering quality compared with the original feature space.

## Dataset

This project uses the Product Classification and Clustering dataset from the UCI Machine Learning Repository.

The notebook automatically downloads the dataset when needed.

## Methodology

### 1. Exploratory Data Analysis

The dataset is inspected to understand its structure and the distribution of products across categories and merchants.

### 2. Feature Engineering

`Merchant ID` and `Category ID` are treated as categorical features and transformed using one-hot encoding.

The resulting feature matrix contains:

- 35,311 products
- 316 encoded features

### 3. Dimensionality Reduction

The encoded features are standardized before applying PCA.

PCA is used for dimensionality analysis and visualization.

### 4. K-Means Baseline

K-Means is first applied to the original standardized feature space.

Best result:

- K = 3
- Silhouette Score = -0.0323

This indicates that the original feature representation does not produce well-separated clusters.

### 5. Autoencoder Representation Learning

An Autoencoder is trained to learn a compact representation of the 316-dimensional feature space.

Architecture:

```text
316 → 128 → 64 → 16 → 64 → 128 → 316
```

The 16-dimensional latent representation is then used for downstream clustering.

### 6. K-Means on Latent Space

K-Means is applied to the Autoencoder latent representation.

Best result:

- K = 2
- Silhouette Score = 0.7475

This represents a substantial improvement over the original feature space.

### 7. DBSCAN

DBSCAN is also applied to the 16-dimensional latent representation.

Results:

- 265 clusters
- 927 noise points
- Silhouette Score = 0.8458

DBSCAN identifies a more fine-grained density-based structure than K-Means.

## Results

| Method | Feature Space | Clusters | Silhouette Score |
|---|---|---:|---:|
| K-Means | Original | 3 | -0.0323 |
| K-Means + Autoencoder | 16D latent space | 2 | 0.7475 |
| DBSCAN + Autoencoder | 16D latent space | 265 | 0.8458 |

The results show that the Autoencoder-based latent representation substantially improves cluster separation compared with the original feature space.

DBSCAN achieves the highest Silhouette Score, while producing a more fine-grained clustering structure.

## Project Structure

```text
product-classification-clustering/
├── notebooks/
│   └── product_classification_clustering.ipynb
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

This project uses Python 3.13.

### 1. Create the virtual environment

```bash
uv venv .venv --python 3.13
```

### 2. Install dependencies

```bash
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
```

### 3. Register the Jupyter kernel

```bash
.\.venv\Scripts\python.exe -m ipykernel install --user --name product-classification-clustering --display-name "Product Classification Clustering (Python 3.13)"
```

### 4. Run the notebook

Open:

```text
notebooks/product_classification_clustering.ipynb
```

Select the Jupyter kernel:

```text
Product Classification Clustering (Python 3.13)
```

Then run the notebook cells sequentially.

## Requirements

The required Python packages are listed in `requirements.txt`:

- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow
- IPykernel

## Technologies

- Python 3.13
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow / Keras
- Jupyter Notebook
- uv