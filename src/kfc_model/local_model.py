

from typing import Dict, Self, Type
from sklearn import clone
from sklearn.base import BaseEstimator
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor

from kfc_model.typing import ArrayLike1D, ArrayLike2D


class LocalModel:
    """
    Wrapper that fits one base model per cluster and predicts using cluster assignment.
    """

    def __init__(
        self,
        base_model: BaseEstimator,
        n_clusters: int
    ):
        self.base_model = base_model
        self.n_clusters = n_clusters
        self.models_: Dict[str, BaseEstimator] = {} # store as lm0, lm1, ...
    
    def fit(
        self,
        X: ArrayLike2D,
        y: ArrayLike1D,
        cluster_labels: ArrayLike1D
    ) -> Self:
        """Fit one model per cluster"""
        self.models_ = {}
        for k in range(self.n_clusters):
            mask = (cluster_labels == k)
            Xk, yk = X[mask], y[mask]
            model = clone(self.base_model)

            if len(yk) == 0: # fallback: fit on all data if cluster is empty
                model.fit(X, y)
            else:
                model.fit(Xk, yk)
            
            if hasattr(model, "classes_"):
                model.n_classes_ = len(model.classes_)

            self.models_[f'lm{k}'] = model
        return self

    def predict(
        self,
        X: ArrayLike2D,
        cluster_labels: ArrayLike1D
    ) -> ArrayLike1D:
        """Predict using the right model per cluster."""
        y_pred = np.zeros(X.shape[0])

        for k, model in self.models_.items():
            cluster_idx = int(k.replace('lm', ''))
            mask = (cluster_labels == cluster_idx)

            if np.any(mask):
                y_pred[mask] = model.predict(X[mask])
        return y_pred
    
    def predict_proba(
        self,
        X: ArrayLike2D,
        cluster_labels: ArrayLike1D
    ) -> ArrayLike2D:
        n_samples = X.shape[0]
        # assume all models have the same number of classes
        n_classes = self.models_['lm0'].n_classes_ if hasattr(self.models_['lm0'], 'n_classes_') else None
        all_probas = []

        for k, model in self.models_.items():
            cluster_idx = int(k.replace('lm', ''))
            mask = (cluster_labels == cluster_idx)
            if np.any(mask):
                probas = model.predict_proba(X[mask])
                all_probas.append((mask, probas))
        
        # combine results into a single array of shape (n_samples, n_classes)
        final_probas = np.zeros((n_samples, n_classes))
        for mask, probas in all_probas:
            final_probas[mask] = probas
        return final_probas

class LocalModelFactory:
    _registry: Dict[str, Type[BaseEstimator]] = {}

    @classmethod
    def register(cls, name: str, model_class: Type[BaseEstimator]):
        """Register a new sklearn model type."""
        if not issubclass(model_class, BaseEstimator):
            raise ValueError("model_class must inherit from sklearn BaseEstimator")
        cls._registry[name.lower()] = model_class

    @classmethod
    def create(cls, name: str, n_clusters: int, **kwargs) -> LocalModel:
        """Return a LocalModel wrapper around the base model."""
        name = name.lower()
        if name not in cls._registry:
            raise ValueError(f"Unknown model type: {name}")
        base_model = cls._registry[name](**kwargs)
        return LocalModel(base_model=base_model, n_clusters=n_clusters)

LocalModelFactory.register("linear", LinearRegression)
LocalModelFactory.register("decision_tree", DecisionTreeRegressor)
LocalModelFactory.register("tree", DecisionTreeRegressor)

LocalModelFactory.register("logistic", LogisticRegression)