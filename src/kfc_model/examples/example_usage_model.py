"""
Full usage of KFCModel for regression and classification with accuracy metrics.
"""

from sklearn.datasets import make_regression, make_classification
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.model_selection import train_test_split
from kfc_model.model import KFCModel
import numpy as np


def regression_example():
    print("=== Regression Example ===")
    # Generate regression data
    X, y = make_regression(
        n_samples=2000,
        n_features=10,
        n_informative=5,
        noise=20,
        random_state=42
    )

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize KFCModel with multiple divergences
    reg_model = KFCModel(
        divergence=[
            'euclidean',
            {'name': 'gkl', 'n_init': 50}
        ],
        local_model='linear',   # linear regression
        combiner='mean'
    )

    # Fit full KFC pipeline
    reg_model.fit(X_train, y_train)

    # Predict on test set
    y_pred = reg_model.predict(X_test)

    # Accuracy metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("Combined predictions (first 10):", y_pred[:10])
    print(f"R² score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")

    print("\nClusters:")
    for name, labels in reg_model.clusters_.items():
        print(f"  {name}: {np.unique(labels, return_counts=True)}")


def classification_example():
    print("\n=== Classification Example ===")
    # Generate classification data
    X, y = make_classification(
        n_samples=2000,
        n_features=10,
        n_informative=5,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42
    )

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize KFCModel for classification
    clf_model = KFCModel(
        divergence=['euclidean', 'gkl'],
        local_model='logistic',   # logistic regression
        combiner='mean'
    )

    # Fit full KFC pipeline
    clf_model.fit(X_train, y_train)

    # Predict classes and probabilities
    y_pred = clf_model.predict(X_test)
    y_proba = clf_model.predict_proba(X_test)

    y_pred_classes = np.argmax(y_proba, axis=1)
    
    # Accuracy metrics
    acc = accuracy_score(y_test, y_pred_classes)
    print("Predicted classes (first 10):", y_pred[:10])
    print("Predicted probabilities (first 3 samples):\n", y_proba[:3])
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes))
    print("\nLog Loss:", log_loss(y_test, y_proba))

    print("\nClusters:")
    for name, labels in clf_model.clusters_.items():
        print(f"  {name}: {np.unique(labels, return_counts=True)}")


if __name__ == "__main__":
    regression_example()
    classification_example()