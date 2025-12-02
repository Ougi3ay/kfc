# KFC-Model: A Python Implementation of the KFC Procedure

This package implements the **KFC Procedure (K-means, Fitting, Combining)** for regression and classification, as proposed by *Sothea Has, Aurélie Fischer, and Mathilde Mougeot (2021)*.

The method is a powerful **cluster-wise supervised learning algorithm**. It works in three steps:

---

## The KFC Procedure

### **K (K-means)**
The training data is partitioned into *K* clusters using one or more clustering algorithms.  
The original paper suggests using K-means with different **Bregman divergences** to create *M* different partitions of the data.

### **F (Fitting)**
A simple, local predictive model (e.g., `LinearRegression` or `LogisticRegression`) is fitted to the data within each cluster.  
This results in *M "candidate models"*, where each candidate model is a collection of *K local models*.

### **C (Combining)**
To make a prediction on a new data point:
1. Each of the *M* candidate models produces a prediction.  
2. The new point is assigned to its nearest cluster in that partition, and the corresponding local model is used.  
3. The *M* predictions are then combined (e.g., by averaging for regression or voting for classification) to produce a final, robust prediction.

---

## Package Structure

kfc_model/
│
├── base.py # Abstract BaseKFC estimator (fit & predict logic)
├── regressor.py # KFCRegressor for regression tasks
├── classifier.py # KFCClassifier for classification tasks
├── combiners.py # BaseCombiner and concrete strategies (MeanCombiner, VotingCombiner)
└── utils/ # Helper functions


---

## Installation

```bash

# for local testing
python -m build
pip install dist/*.whl

pip install kfc


from kfc_model.regressor import KFCRegressor
from kfc_model.combiners import MeanCombiner
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_regression

# 1. Define the components for the KFC model
# K: A list of M clusterers. Here, just M=1 for simplicity.
clusterers = [KMeans(n_clusters=5, random_state=42, n_init=10)]

# F: The local model to fit on each cluster
local_model = LinearRegression()

# C: The strategy to combine predictions from the M clusterers
combiner = MeanCombiner()

# 2. Create the KFCRegressor
kfc_reg = KFCRegressor(
    clusterers=clusterers,
    local_estimator=local_model,
    combiner=combiner
)

# 3. Fit the model
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, noise=10, random_state=42)
kfc_reg.fit(X, y)

# 4. Make predictions
predictions = kfc_reg.predict(X[:5])
print(predictions)



## **Reference:**

Has, S., Fischer, A., & Mougeot, M. (2021). KFC: A Cluster-wise Supervised Learning Procedure for Regression and Classification.