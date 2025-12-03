from typing import Any, Dict, List, Self, Union, Optional
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from kfc_model.bregman import KMeansBregman
from kfc_model.local_model import LocalModelFactory
from kfc_model.combiner import CombinerFactory
from kfc_model.typing import ArrayLike1D, ArrayLike2D, DivergenceConfig, DivergenceSpec


class KFCModel(BaseEstimator):
    """
    KFCModel
    --------
    A unified regression and classification model using the (K-step, F-step, C-step) pipeline.

    The KFC pipeline consists of three stages:

    1. **K-step (Clustering)**:
    - Features of the input dataset are clustered using one or more Bregman divergences:
        + `KMeansBregman` algorithm (`'euclidean'`, `'gkl'`, `'logistic'`, `'itakura'`, `'exponential'`, `'polynomial'`, etc.)
    - Each divergence configuration produces a set of cluster assignments for the data.
    - Clusters are stored in `self.clusters_` and KMeans models in `self.kmeans_models_`.

    2. **F-step (Fitting local models)**:
    - For each cluster from each divergence, a local model is trained on the corresponding subset of data.
        + Scikit-Learn models (`LinearRegression`, `DecisionTreeRegressor`, `LogisticRegression`, etc.) can be used.
    - The local models are wrapped in `LocalModel` instances and stored in `self.candidate_models_`.
    - Each cluster has its own tailored prediction model.

    3. **C-step (Combining predictions)**:
    - Candidate predictions from all divergences and clusters are aggregated using a combiner:
        + Combiner strategies: `'mean'` for regression, `'voting'` for classification.
    - The combiner is created via `CombinerFactory` and stored in `self.combiner`.
    - Final combined predictions are returned by `predict` or `predict_proba`.

    Parameters
    ----------
    divergence : str or list of dict, default="euclidean"
        The divergence(s) used for clustering. Can be a single string or a list of
        divergence specifications. Each dict may include parameters like `'n_clusters'`, `'n_init'`, etc.

    local_model_name : str, default="linear"
        Name of the base model for local fitting. Must be registered in `LocalModelFactory`.
        Common options: `'linear'`, `'decision_tree'`, `'logistic'`.

    local_model_params : dict, optional
        Additional keyword arguments passed to the local model constructor.

    combiner_name : str, default="mean"
        Name of the combiner strategy used to merge candidate model predictions.
        Examples: `'mean'` (regression), `'voting'` (classification).

    combiner_params : dict, optional
        Additional keyword arguments passed to the combiner constructor.

    Attributes
    ----------
    **clusters_** : dict
        Cluster labels for each divergence after K-step. Keys are divergence names (e.g., `'BD1'`).

    **kmeans_models_** : dict
        Fitted `KMeansBregman` models for each divergence.

    **candidate_models_** : dict
        Local models trained for each cluster of each divergence.

    **combiner** : BaseCombiner
        The combiner used to aggregate candidate predictions into final predictions.

    **combined_predictions_** : np.ndarray
        Final predictions stored after running `c_step` with `store_result=True`.

    Methods
    -------
    fit(X, y)
        Fit the entire KFC pipeline (K-step, F-step, C-step) on the training data.

    predict(X)
        Return combined predictions for input X.

    predict_proba(X)
        Return combined class probabilities (for classification tasks).

    k_step(X)
        Perform clustering on the features of X using configured divergences.

    f_step(X, y)
        Train local models for each cluster of each divergence.

    c_step(X, store_result=True)
        Combine predictions of all candidate models for X using the combiner.

    _predict_candidates(X, proba=False)
        Generate predictions from all candidate models without combining them.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import train_test_split
    >>> from kfc_model.model import KFCModel
    >>> from sklearn.datasets import make_regression
    >>> 
    >>> # Generate synthetic regression data
    >>> X, y = make_regression(n_samples=2000, n_features=10, n_informative=5, noise=20, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    >>> 
    >>> # Initialize the KFC model
    >>> model = KFCModel(
    ...     divergence=['euclidean', {'name': 'gkl', 'n_init': 50}],
    ...     local_model_name='linear',
    ...     combiner_name='mean'
    ... )
    >>> 
    >>> # Fit the model
    >>> model.fit(X_train, y_train)
    KFCModel(...)
    >>> 
    >>> # Predict on test data
    >>> y_pred = model.predict(X_test)
    >>> y_pred.shape
    (400,)
    >>> 
    >>> # Inspect cluster assignments
    >>> list(model.clusters_.keys())
    ['BD1', 'BD2']
    >>> np.unique(model.clusters_['BD1'], return_counts=True)
    (array([0, 1, 2]), array([135, 133, 132]))
    >>> 
    >>> # Access combined predictions (stored after fitting)
    >>> model.combined_predictions_.shape
    (800,)
    >>> 
    >>> # Access candidate local models
    >>> list(model.candidate_models_.keys())
    ['BD1', 'BD2']
    >>> list(model.candidate_models_['BD1'].keys())
    ['lm0', 'lm1', 'lm2']
    >>> 
    >>> # Access combiner object
    >>> model.combiner
    <kfc_model.combiner.BaseCombiner object at 0x...>

    Notes
    -----
    - The KFC pipeline allows multiple divergences and local models to be used simultaneously, providing a flexible ensemble-like approach.
    - Clustering and local fitting are performed on a split of the training data (`X_pre`), while the combining step can be done on the remaining data (`X_agg`).
    """
    
    def __init__(
        self,
        divergence: Union[str, List[DivergenceSpec]] = "euclidean",
        local_model_name: str = "linear",
        local_model_params: Optional[Dict[str, Any]] = None,
        combiner_name: str = "mean",
        combiner_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize KFCModel parameters."""
        self.divergence: Union[str, List[DivergenceSpec]] = divergence
        self.local_model_name: str = local_model_name
        self.local_model_params: Dict[str, Any] = local_model_params or {}
        self.combiner_name: str = combiner_name
        self.combiner_params: Dict[str, Any] = combiner_params or {}

        self.combiner = None
        self.clusters_: Dict[str, ArrayLike1D] = {}
        self.kmeans_models_: Dict[str, Any] = {}
        self.candidate_models_: Dict[str, Dict[str, Any]] = {}
        self._divergence_cfgs: List[DivergenceConfig] = []

        self._configure_divergence()

    def _configure_divergence(self) -> None:
        """Normalize divergence parameter into a list of configurations."""
        default_params: DivergenceConfig = {
            "n_clusters": 3,
            "max_iter": 300,
            "n_init": 10,
            "tol": 1e-6,
            "random_state": None,
        }

        if isinstance(self.divergence, str):
            self._divergence_cfgs = [{"name": self.divergence, **default_params}]
        elif isinstance(self.divergence, list):
            configs: List[DivergenceConfig] = []
            for item in self.divergence:
                if isinstance(item, str):
                    configs.append({"name": item, **default_params})
                elif isinstance(item, dict) and "name" in item:
                    configs.append({**default_params, **item})
                else:
                    raise TypeError(
                        "Each divergence must be a string or a dict with 'name'."
                    )
            self._divergence_cfgs = configs
        else:
            raise TypeError("divergence must be a string or a list of strings/dicts.")

    def fit(self, X: ArrayLike2D, y: ArrayLike1D) -> Self:
        """Fit the full KFC pipeline (K-step, F-step, C-step)."""
        X_pre, X_agg, y_pre, y_agg = train_test_split(
            X, y, test_size=0.5, random_state=42
        )
        self.X_pre, self.X_agg = X_pre, X_agg
        self.y_pre, self.y_agg = y_pre, y_agg

        self.k_step(X_pre)
        self.f_step(X_pre, y_pre)
        
        # c-step
        # First: use X_agg to get candidates (sample, M)
        self.candidate_predict_ = self._predict_candidates(X_agg) 

        # Create combiner if not exists
        self.combiner = CombinerFactory.create(self.combiner_name, **self.combiner_params)
        
        # train 
        if hasattr(self.combiner, 'fit'):
            self.combiner.fit(self.candidate_predict_, y_agg)
        return self

    def _predict_candidates(
        self, X: ArrayLike2D, proba: bool = False
    ) -> np.ndarray:
        """Predict using all candidate models."""
        n_samples = X.shape[0]
        n_divs = len(self.candidate_models_)
        if proba:
            n_classes = self.candidate_models_['BD1']['lm0'].n_classes_
            result = np.zeros((n_samples, n_classes, n_divs))
        else:
            result = np.zeros((n_samples, n_divs))

        for i, (div_name, F_k) in enumerate(self.candidate_models_.items()):
            kmeans = self.kmeans_models_[div_name]
            labels = kmeans.predict(X)

            for cluster_idx, lm in F_k.items():
                cluster_number = int(cluster_idx.replace('lm', ''))
                mask = (labels == cluster_number)
                if np.any(mask):
                    if proba:
                        result[mask, :, i] = lm.predict_proba(X[mask])
                    else:
                        result[mask, i] = lm.predict(X[mask])
        return result


    def predict(self, X: ArrayLike2D) -> ArrayLike1D:
        """Predict final output for X."""
        if not self.candidate_models_:
            raise ValueError("Model is not fitted yet.")
        
        if not hasattr(self, "combiner") or self.combiner is None:
            raise ValueError("Combiner is not fitted. Call fit() first.")
        
        candidate_preds = self._predict_candidates(X)
        final_pred = self.combiner.combine(candidate_preds)
        return final_pred

    def predict_proba(self, X: ArrayLike2D) -> ArrayLike2D:
        """Predict probability for classification tasks."""
        if not self.candidate_models_:
            raise ValueError("Model is not fitted yet.")
        
        if not hasattr(self, "combiner") or self.combiner is None:
            raise ValueError("Combiner is not fitted. Call fit() first.")
        
        all_probas = self._predict_candidates(X, proba=True)
        return self.combiner.combine_proba(all_probas)

    # Helper function
    def k_step(self, X: ArrayLike2D) -> None:
        """K-step: cluster features using KMeansBregman for each divergence."""
        self.clusters_ = {}
        self.kmeans_models_ = {}

        for idx, cfg in enumerate(self._divergence_cfgs):
            kmeans = KMeansBregman(
                n_clusters=cfg['n_clusters'],
                divergence=cfg['name'],
                max_iter=cfg.get('max_iter', 300),
                n_init=cfg.get('n_init', 10),
                tol=cfg.get('tol', 1e-6),
                random_state=cfg.get('random_state', None),
                **{
                    k: v
                    for k, v in cfg.items()
                    if k
                    not in ['name', 'n_clusters', 'max_iter', 'n_init', 'tol', 'random_state']
                },
            )
            kmeans.fit(X)
            key = f"BD{idx+1}"
            self.clusters_[key] = kmeans.labels_
            self.kmeans_models_[key] = kmeans

    def f_step(self, X: ArrayLike2D, y: ArrayLike1D) -> None:
        """F-step: train local models for each cluster of each divergence."""
        if not self.clusters_:
            raise ValueError("Run k_step first.")

        self.candidate_models_ = {}
        for div_name, labels in self.clusters_.items():
            n_clusters = len(np.unique(labels))
            local_model = LocalModelFactory.create(
                self.local_model_name, n_clusters=n_clusters, **self.local_model_params
            )
            local_model.fit(X, y, labels)
            self.candidate_models_[div_name] = local_model.models_