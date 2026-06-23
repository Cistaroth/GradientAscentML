"""
Parity check: custom LogisticRegression vs scikit-learn (unregularized).

Fits both implementations on the same data and reports how closely their
weights and predictions agree. Because the custom model is unregularized,
sklearn is configured to be unregularized too (C=inf), so the two solve the
*same* optimization problem and should converge to the same optimum.

Run:
    python -m src.logistic_regression.compare_sklearn
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as SkLogisticRegression

from src.logistic_regression.model import LogisticRegression

# Tolerance the benchmark uses for its parity assertions.
TOLERANCE = 1e-5


def load_binary_iris() -> tuple[np.ndarray, np.ndarray]:
    """Load iris, keep the (non-separable) versicolor/virginica pair, remap to {0, 1}."""
    data = load_iris(as_frame=True)
    X, y = data.data, data.target
    mask = y.isin([1, 2])
    X = X[mask].to_numpy()
    y = (y[mask] == 2).astype(int).to_numpy()
    return X, y


def fit_sklearn(X: np.ndarray, y: np.ndarray) -> SkLogisticRegression:
    """Fit an unregularized sklearn model, tolerant to API changes across versions."""
    kwargs = dict(solver="lbfgs", max_iter=10000, tol=1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return SkLogisticRegression(C=np.inf, **kwargs).fit(X, y)
        except (TypeError, ValueError):
            # Older sklearn: no C=inf path -> fall back to penalty=None.
            return SkLogisticRegression(penalty=None, **kwargs).fit(X, y)


def sklearn_weights(model: SkLogisticRegression) -> np.ndarray:
    """Stack sklearn's [intercept, coef...] to match the custom model's layout."""
    return np.concatenate([model.intercept_, model.coef_[0]])


def compare(label: str, X: np.ndarray, y: np.ndarray, tolerance: float = TOLERANCE) -> bool:
    """Fit both models on identical inputs and report agreement. Returns pass/fail."""
    mine = LogisticRegression().fit(X, y)
    skl = fit_sklearn(X, y)

    w_mine = mine.weights
    w_skl = sklearn_weights(skl)
    weight_gap = np.abs(w_mine - w_skl).max()

    # Predictions on the same inputs.
    proba_mine = mine.predict_proba(X)
    proba_skl = skl.predict_proba(X)[:, 1]
    proba_gap = np.abs(proba_mine - proba_skl).max()
    label_agreement = np.mean(mine.predict(X) == skl.predict(X))

    weights_ok = weight_gap <= tolerance
    verdict = "PASS" if weights_ok else "FAIL"

    print(f"\n=== {label} ===")
    with np.printoptions(precision=6, suppress=True, linewidth=120):
        print(f"  custom weights : {w_mine}")
        print(f"  sklearn weights: {w_skl}")
    print(f"  max |Δ weight|        : {weight_gap:.3e}   (tolerance {tolerance:.0e}) -> {verdict}")
    print(f"  max |Δ probability|   : {proba_gap:.3e}")
    print(f"  label agreement       : {label_agreement:.4%}")
    return weights_ok


def main() -> None:
    X, y = load_binary_iris()
    print("Comparing custom LogisticRegression vs scikit-learn (both unregularized)")
    print(f"Data: iris versicolor-vs-virginica, X={X.shape}, positives={int(y.sum())}/{len(y)}")

    # Raw features: near-separable + ill-conditioned -> solvers stop at slightly
    # different points, so raw weights typically miss a 1e-5 tolerance.
    raw_ok = compare("Raw features", X, y)

    # Standard-scaled features: well-conditioned -> tight agreement. This mirrors
    # what the benchmark pipeline actually feeds the model.
    Xs = StandardScaler().fit_transform(X)
    scaled_ok = compare("Standard-scaled features", Xs, y)

    print("\n" + "-" * 60)
    print(f"Raw features within {TOLERANCE:.0e}    : {raw_ok}")
    print(f"Scaled features within {TOLERANCE:.0e} : {scaled_ok}")
    print("-" * 60)
    if scaled_ok:
        print("Implementations match to solver precision on conditioned inputs.")
    else:
        print("Mismatch persists after scaling -> investigate (likely a real bug).")


if __name__ == "__main__":
    main()