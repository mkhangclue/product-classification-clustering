# Product Clustering & Representation Learning

An unsupervised machine learning project that explores product segmentation using K-Means, Autoencoder-based representation learning, and DBSCAN.

## Project Overview

This project analyzes product data using unsupervised machine learning techniques to identify meaningful product groupings.

The main objective is to evaluate whether **Autoencoder-based representation learning** can improve clustering quality compared with clustering directly on the original feature space.

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

## Dataset

This project uses the **Product Classification and Clustering dataset** from the UCI Machine Learning Repository.

The notebook automatically downloads the dataset when needed.

## Methodology

### 1. Exploratory Data Analysis

The dataset is inspected to understand its structure and the distribution of products across categories and merchants.

### 2. Feature Engineering

`Merchant ID` and `Category ID` are treated as categorical features and transformed using one-hot encoding.

The resulting feature matrix contains:

- 35,311 products
- 316 encoded features

### 3. Feature Standardization and PCA

The encoded features are standardized before applying PCA.

PCA is used to analyze the dimensionality of the feature space and support visualization.

### 4. K-Means Baseline

K-Means is first applied to the original standardized feature space to establish a baseline.

**Best result:**

- K = 3
- Silhouette Score = -0.0323

The negative Silhouette Score indicates that the original feature representation does not produce well-separated clusters.

### 5. Autoencoder Representation Learning

An Autoencoder is trained to learn a compact representation of the 316-dimensional feature space.

Architecture:

```text
316 → 128 → 64 → 16 → 64 → 128 → 316
```

The 16-dimensional latent representation is then used for downstream clustering.

### 6. K-Means on Latent Representation

K-Means is applied to the 16-dimensional latent representation learned by the Autoencoder.

**Best result:**

- K = 2
- Silhouette Score = 0.7475

Compared with the baseline, the latent representation produces substantially better cluster separation.

### 7. DBSCAN on Latent Representation

DBSCAN is applied to the same 16-dimensional latent representation.

**Results:**

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

The results show that the learned latent representation substantially improves cluster separation compared with the original feature space.

DBSCAN achieves the highest Silhouette Score, while producing a much more fine-grained clustering structure than K-Means.

However, a higher Silhouette Score does not automatically mean that DBSCAN produces the most useful business segmentation. The number of clusters, noise points, cluster interpretability, and business relevance should also be considered.

## Key Findings

1. Clustering directly on the original feature space produced poor separation, with a negative Silhouette Score.
2. The Autoencoder successfully learned a lower-dimensional representation that improved clustering performance.
3. K-Means on the latent representation achieved a Silhouette Score of 0.7475.
4. DBSCAN achieved the highest Silhouette Score of 0.8458 but generated 265 clusters and 927 noise points.
5. Therefore, clustering quality should be evaluated using both numerical metrics and the interpretability of the resulting clusters.

## Limitations

- The clustering results are based on the available product attributes and may not capture all real-world factors influencing product similarity.
- Silhouette Score alone is insufficient to determine the business usefulness of a clustering solution.
- The high number of DBSCAN clusters may make the resulting segmentation difficult to interpret or apply in practice.
- No external business labels are available to independently validate the discovered clusters.

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
