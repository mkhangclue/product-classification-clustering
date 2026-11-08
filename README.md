# Product Clustering & Representation Learning

An unsupervised machine learning project that explores product clustering using K-Means, Autoencoder-based representation learning, and DBSCAN.

## Project Overview

This project analyzes product data using unsupervised machine learning techniques to discover patterns and groupings among products.

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

The dataset contains 35,311 product records with 7 original columns.

## Methodology

### 1. Exploratory Data Analysis

The dataset is inspected to understand its structure, data types, missing values, and the distribution of products across categories and merchants.

### 2. Feature Engineering

`Merchant ID` and `Category ID` are selected as input features and treated as categorical variables.

One-hot encoding is applied to transform these features into a numerical representation.

The resulting feature matrix contains:

- 35,311 product records
- 316 encoded features

### 3. Feature Standardization and PCA

The encoded features are standardized before applying PCA.

PCA is used to analyze the dimensionality of the feature space and support two-dimensional visualization.

The first two principal components explain approximately 1.23% of the total variance. Therefore, the two-dimensional PCA representation is used for visualization rather than as the primary input for clustering.

### 4. K-Means Baseline

K-Means is first applied to the original standardized feature space to establish a baseline.

**Best result:**

- K = 3
- Silhouette Score = -0.0323

The negative Silhouette Score indicates poor cluster separation in the original standardized feature space.

### 5. Autoencoder Representation Learning

An Autoencoder is trained to learn a compact representation of the original 316-dimensional feature space.

Architecture:

```text
316 → 128 → 64 → 16 → 64 → 128 → 316
```

The 16-dimensional latent representation is extracted from the trained encoder and used for downstream clustering.

### 6. K-Means on Latent Representation

K-Means is applied to the 16-dimensional latent representation learned by the Autoencoder.

**Best result:**

- K = 2
- Silhouette Score = 0.7588

Compared with the baseline, clustering on the latent representation achieves substantially better cluster separation.

### 7. DBSCAN on Latent Representation

DBSCAN is applied to the same 16-dimensional latent representation.

**Results:**

- 212 clusters
- 633 noise points
- Silhouette Score = 0.6622

The Silhouette Score is calculated using only observations that are not classified as noise.

DBSCAN identifies a more fine-grained density-based structure, although the large number of clusters may make the results more difficult to interpret.

## Results

| Method | Feature Space | Clusters | Noise Points | Silhouette Score |
|---|---|---:|---:|---:|
| K-Means | Original (316D) | 3 | 0 | -0.0323 |
| K-Means + Autoencoder | Latent (16D) | 2 | 0 | **0.7588** |
| DBSCAN + Autoencoder | Latent (16D) | 212 | 633 | 0.6622 |

**Note:** DBSCAN's Silhouette Score is calculated on non-noise observations only, while K-Means scores are calculated using the entire dataset. Therefore, the scores are not directly comparable.

## Cluster Analysis

K-Means on the Autoencoder latent representation identifies two clusters with an imbalanced distribution.

| Cluster | Product Count | Percentage |
|---|---:|---:|
| Cluster 0 | 33,215 | 94.06% |
| Cluster 1 | 2,096 | 5.94% |

### Category Distribution

The two clusters show different category distributions.

**Cluster 0** has relatively higher proportions of:

- Fridge Freezers: 16.55%
- Washing Machines: 12.08%
- Mobile Phones: 11.42%
- Fridges: 10.78%
- CPUs: 10.30%

**Cluster 1** is more concentrated in:

- TVs: 29.91%
- Digital Cameras: 23.38%
- CPUs: 21.09%
- Mobile Phones: 13.69%
- Microwaves: 9.35%

These differences suggest that the learned representation captures variations in the distribution of product categories across clusters.

### Merchant Distribution

The most frequent merchants also differ between the two clusters.

- Cluster 0: Merchant 3 has the highest product count (2,547).
- Cluster 1: Merchant 73 has the highest product count (313).

These observations provide additional context for interpreting the discovered clusters.

## Key Findings

1. K-Means applied directly to the original feature space produced poor cluster separation, with a negative Silhouette Score.
2. Autoencoder-based representation learning substantially improved the clustering structure in the 16-dimensional latent space.
3. K-Means on the latent representation achieved the highest Silhouette Score among the evaluated methods (0.7588).
4. DBSCAN identified 212 density-based clusters and 633 noise points, with a Silhouette Score of 0.6622 on non-noise observations.
5. The K-Means latent clusters showed different Category and Merchant distributions, but their sizes were highly imbalanced.

Overall, K-Means on the Autoencoder latent representation provided the strongest clustering result according to the Silhouette Score in these experiments.

## Limitations

- The clustering model uses only `Merchant ID` and `Category ID` as input features. It does not capture richer product information such as product names, descriptions, prices, or other product attributes.
- The resulting clusters reflect patterns in merchant and category relationships rather than comprehensive product similarity or customer purchasing behavior.
- The two K-Means clusters are highly imbalanced, with Cluster 0 containing 94.06% of the observations.
- DBSCAN produces a large number of clusters, which may reduce interpretability.
- DBSCAN's Silhouette Score is calculated only on non-noise observations, limiting direct comparison with K-Means.
- No external labels are available to independently validate the discovered clusters.

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
