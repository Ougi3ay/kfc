"""Divergence base"""
from typing import Any, Dict, Type
import numpy as np

from kfc_model.typing import ArrayLike1D, ArrayLike2D

class Divergence:
    """Base class for divergence functions."""

    def __init__(self, deg: int = 3) -> None:
        self.deg = deg

    def compute(self, X: ArrayLike2D, y: ArrayLike1D) -> ArrayLike1D:
        """
        Compute divergence between x and y.

        Parameters
        ----------
        x : Array2D
            Input data of shape (n_samples, n_features)
        y : Array1D
            Reference vector of shape (n_features,)

        Returns
        -------
        Array1D
            Divergence values for each row in x.
        """
        raise NotImplementedError("Subclasses must implement this method.")

class EuclideanDivergence(Divergence):
    """
    Compute the squared Euclidean divergence (distance) between each row of X and a vector y.

    The squared Euclidean divergence is defined as:
        D(X, y) = Σ (X - y)²

    where the summation is taken element-wise along each row of X.
    This is equivalent to the squared L2 norm between each row of X and y.
    """

    def compute(self, X: ArrayLike2D, y: ArrayLike1D) -> ArrayLike1D:
        """
        Compute row-wise squared Euclidean divergences.

        Parameters
        ----------
        x : ArrayLike2D
            Input 2D array of shape (n_samples, n_features).
        y : ArrayLike1D
            Reference vector of shape (n_features,).

        Returns
        -------
        ArrayLike1D
            1D array of squared Euclidean distances for each row in X.
        """
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)

        res = X - y
        return np.sum(res ** 2, axis=1)

class GKLDivergence(Divergence):
    """
    Compute the row-wise Generalized Kullback–Leibler (GKL) divergence
    between each row of X and a target vector y.

    The GKL divergence is defined as:
        D(X || y) = Σ [ X * log(X / y) - (X - y) ]

    where the summation is applied element-wise along each row of X.
    """

    def compute(self, X: ArrayLike2D, y: ArrayLike1D) -> ArrayLike1D:
        """
        Compute GKL divergence for each row of X with respect to vector y.

        Parameters
        ----------
        x : ArrayLike2D
            Input 2D array of shape (n_samples, n_features).
        y : ArrayLike1D
            1D reference vector of shape (n_features,).

        Returns
        -------
        ArrayLike1D
            1D array containing GKL divergence values for each row of X.
        """
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)

        # Clip to avoid log(0) or negative
        X = np.clip(X, 1e-10, None)
        y = np.clip(y, 1e-10, None)

        div = X / y
        sub = X - y

        return np.sum(X * np.log(div) - sub, axis=1)

class LogisticDivergence(Divergence):
    """
    Compute the row-wise binary Kullback–Leibler (logistic) divergence
    between each row of X and a reference vector y.

    The logistic (binary KL) divergence is defined as:
        D(X || y) = Σ [ x * log(x / y) + (1 - x) * log((1 - x) / (1 - y)) ]

    where X and y have values in the interval [0, 1].
    """

    def compute(self, X: ArrayLike2D, y: ArrayLike1D) -> ArrayLike1D:
        """
        Compute binary KL (logistic) divergence for each row of X relative to y.

        Parameters
        ----------
        x : ArrayLike2D
            Input 2D array of shape (n_samples, n_features), values ∈ [0, 1].
        y : ArrayLike1D
            1D reference vector of shape (n_features,), values ∈ [0, 1].

        Returns
        -------
        ArrayLike1D
            1D array of divergence values for each row of X.
        """
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)

        # Avoid division by zero or log(0)
        eps = 1e-12
        X = np.clip(X, eps, 1 - eps)
        y = np.clip(y, eps, 1 - eps)

        res1 = X / y
        res2 = (1 - X) / (1 - y)

        return np.sum(X * np.log(res1) + (1 - X) * np.log(res2), axis=1)

class ItakuraDivergence(Divergence):
    """
    Compute the row-wise Itakura–Saito (IS) divergence
    between each row of X and a reference vector y.

    The Itakura–Saito divergence is defined as:
        D(X || y) = Σ [ (X / y) - log(X / y) - 1 ]

    where X and y contain strictly positive values.
    """

    def compute(self, X: ArrayLike2D, y: ArrayLike1D) -> ArrayLike1D:
        """
        Compute Itakura–Saito divergence for each row of X relative to y.

        Parameters
        ----------
        x : ArrayLike2D
            Input 2D array of strictly positive values (n_samples, n_features).
        y : ArrayLike1D
            Reference 1D array of strictly positive values (n_features,).

        Returns
        -------
        ArrayLike1D
            1D array of IS divergence values for each row in X.
        """
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)

        # Avoid division by zero and log(0)
        eps = 1e-12
        X = np.clip(X, eps, None)
        y = np.clip(y, eps, None)

        res = X / y
        return np.sum(res - np.log(res) - 1, axis=1)

class PolynomialDivergence(Divergence):
    """
    Compute the row-wise polynomial divergence between each row of X and a reference vector y.

    The polynomial divergence is defined as:
        For even degree:
            D(X || y) = Σ [ (X^deg - y^deg) - deg * (X - y) * y^(deg-1) ]
        For odd degree:
            D(X || y) = Σ [ (X^deg - y^deg) - deg * (X - y) * sign(y) * y^(deg-1) ]

    where the summation is applied element-wise across each row of X.
    """

    def __init__(self, deg: int = 3) -> None:
        """
        Initialize a polynomial divergence instance.

        Parameters
        ----------
        deg : int, optional
            Degree of the polynomial (default is 3).
        """
        super().__init__(deg=deg)

    def compute(self, X: ArrayLike2D, y: ArrayLike1D) -> ArrayLike1D:
        """
        Compute polynomial divergence for each row of X relative to y.

        Parameters
        ----------
        x : ArrayLike2D
            Input 2D array of shape (n_samples, n_features).
        y : ArrayLike1D
            Reference 1D array of shape (n_features,).

        Returns
        -------
        ArrayLike1D
            1D array of polynomial divergence values for each row in X.
        """
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)
        deg = self.deg

        diff = X - y
        poly_diff = X**deg - y**deg

        if deg % 2 == 0:
            tem = diff * y**(deg - 1)
        else:
            tem = diff * np.sign(y) * y**(deg - 1)

        res = np.sum(poly_diff - deg * tem, axis=1)
        return res

class ExponentialDivergence(Divergence):
    """
    Compute the row-wise exponential-based divergence
    between each row of X and a reference vector y.

    The exponential divergence is defined as:
        D(X || y) = Σ [ exp(X_j) - (1 + X_j - y_j) * exp(y_j) ]

    where the summation is taken element-wise along each row of X.
    """

    def compute(self, X: ArrayLike2D, y: ArrayLike1D) -> ArrayLike1D:
        """
        Compute exponential divergence for each row of X relative to y.

        Parameters
        ----------
        x : ArrayLike2D
            Input 2D array of shape (n_samples, n_features).
        y : ArrayLike1D
            Reference 1D array of shape (n_features,).

        Returns
        -------
        ArrayLike1D
            1D array of exponential divergence values for each row in X.
        """
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)

        exp_y = np.exp(y)
        res = (1 + X - y) * exp_y

        return np.sum(np.exp(X) - res, axis=1)

class DivergenceFactory:
    """Factory class to create divergence objects by name."""
    _registry: Dict[str, Type[Divergence]] = {
        "euclidean": EuclideanDivergence,
        "gkl": GKLDivergence,
        "logistic": LogisticDivergence,
        "itakura": ItakuraDivergence,
        "exponential": ExponentialDivergence,
        "polynomial": PolynomialDivergence,
    }
    @classmethod
    def create(cls, name: str, **kwargs: Any) -> Divergence:
        """
        Create an instance of a divergence class by name.

        Parameters
        ----------
        name : str
            Name of the divergence ('euclidean', 'gkl', 'logistic', etc.).
        **kwargs : dict
            Additional arguments to pass to the divergence constructor.

        Returns
        -------
        Divergence
            An instance of the requested divergence class.
        """
        name = name.lower()
        if name not in cls._registry:
            raise ValueError(f"Unknown divergence type: {name}")
        return cls._registry[name](**kwargs)
