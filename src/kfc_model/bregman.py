"""Bregman Kmean module"""

from typing import Any, Dict, Optional, Self
from sklearn.base import BaseEstimator, ClusterMixin, check_array
from sklearn.dummy import check_random_state
import numpy as np

from kfc_model.divergence import Divergence, DivergenceFactory
from kfc_model.typing import ArrayLike1D, ArrayLike2D

class KMeansBregman(BaseEstimator, ClusterMixin):
    """
    Scikit-learn compatible KMeans using arbitrary Bregman divergences.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        divergence: str = "euclidean",
        max_iter: int = 300,
        n_init: int = 10,
        tol: float = 1e-6,
        random_state: Optional[int] = None,
        **divergence_params: Any
    ) -> None:
        """
        Initialize the KMeansBregman clustering instance.

        Parameters
        ----------
        n_clusters : int
            Number of clusters.
        divergence : str (euclidean | gkl | logistic | itakura | exponential | polynomial)
            Type of Bregman divergence ('euclidean', 'gkl', etc.).
        max_iter : int
            Maximum number of iterations per initialization.
        n_init : int
            Number of random initializations.
        tol : float
            Convergence tolerance for centroid updates.
        random_state : int or None
            Random seed for reproducibility.
        """
        self.n_clusters = n_clusters
        self.divergence = divergence
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.random_state = random_state
        self.divergence_params: Dict[str, Any] = divergence_params

        # Attributes (initialized after fitting)
        self.cluster_centers_: Optional[ArrayLike2D] = None
        self.labels_: Optional[ArrayLike1D] = None
        self.inertia_: Optional[float] = None
        self.divergence_: Optional[Divergence] = None
        
    # fit, predict, fit_predict
    def fit(self, X: ArrayLike2D, y: None = None) -> Self:
        """
        Fit the Bregman KMeans model on data X.

        Parameters
        ----------
        X : ArrayLike2D
            Data of shape (n_samples, n_features).
        y : None
            Ignored; exists for compatibility.

        Returns
        -------
        Self
            The fitted BregmanKMeans instance.
        """
        X = check_array(X, dtype=float)
        rng = check_random_state(self.random_state)

        # build divergence instance
        divergence: Divergence = DivergenceFactory.create(
            self.divergence,
            **self.divergence_params
        )
        self.divergence_ = divergence

        best_inertia: float = np.inf
        best_centers: Optional[ArrayLike2D] = None
        best_labels: Optional[ArrayLike1D] = None

        for _ in range(self.n_init):
            # initialize centroids
            centroids = self._init_centroids(X, rng)

            # K-Means is an iterative algorithm that stops early if it converges.
            for _ in range(self.max_iter):
                # Assign each data point to the nearest centroid, using the chosen Bregman divergence.
                labels = self._assign_clusters(X, centroids, divergence)

                # Recompute the centroids based on the newly assigned cluster points.
                new_centroids = self._update_centroids(X, labels)

                # Measure centroid movement for convergence
                shift = np.sum((new_centroids - centroids) ** 2)

                centroids = new_centroids

                if shift <= self.tol:
                    break
            
            # Comput inertia = sum divergence
            inertia = 0.0
            for k in range(self.n_clusters):
                cluster_points = X[labels == k]
                if cluster_points.size > 0:
                    inertia += float(divergence.compute(cluster_points, centroids[k]).sum())
            
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centroids.copy()
                best_labels = labels.copy()
        
        # store results
        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia

        return self

    def predict(self, X: ArrayLike2D) -> ArrayLike1D:
        """
        Predict the closest cluster for each sample in X.

        Parameters
        ----------
        X : ArrayLike2D
            New data.

        Returns
        -------
        ArrayLike1D
            Cluster indices for each sample.
        """
        if self.cluster_centers_ is None or self.divergence_ is None:
            raise ValueError("Model has not been fitted yet.")
        
        X = check_array(X, dtype=float)
        return self._assign_clusters(X, self.cluster_centers_, self.divergence_)

    def fit_predict(self, X: ArrayLike2D, y: None = None) -> ArrayLike1D:
        """
        Fit the model and return cluster labels for each sample.

        Parameters
        ----------
        X : ArrayLike2D
            Data to cluster.
        y : None
            Ignored.

        Returns
        -------
        ArrayLike1D
            Cluster labels.
        """
        self.fit(X)
        return self.labels_
    

    # Helper function
    def _init_centroids(
        self, 
        X: ArrayLike2D, 
        random_state: np.random.RandomState
    ) -> ArrayLike2D:
        """Randomly select initial centroids from the dataset."""
        n_samples = X.shape[0]
        indices = random_state.choice(n_samples, self.n_clusters, replace=False)
        return X[indices].copy()

    def _assign_clusters(
        self,
        X: ArrayLike2D,
        centroids: ArrayLike2D,
        divergence: Divergence
    ) -> ArrayLike1D:
        """Assign each point in X to the closest centroid based on divergence."""
        # Shape -> (n_samples, n_clusters)
        distances = np.vstack([
            divergence.compute(X, centroids[k]) for k in range(self.n_clusters)
        ]).T
        return np.argmin(distances, axis=1).astype(np.int64)

    def _update_centroids(
        self,
        X: ArrayLike2D,
        labels: ArrayLike1D
    ) -> ArrayLike2D:
        """Recompute centroids as the mean of points assigned to each cluster."""
        new_centroids = np.zeros((self.n_clusters, X.shape[1]), dtype=X.dtype)

        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if cluster_points.size == 0:
                # randomly reinitialize empty cluster
                new_centroids[k] = X[np.random.randint(0, len(X))]
            else:
                new_centroids[k] = cluster_points.mean(axis=0)
        return new_centroids
