from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.logistic_regression.weights_rebasing import LogisticRegressionWeightsRebasing

class LogisticRegression():
    def __init__(
        self,
        weights: np.ndarray | None = None,
        hessian: np.ndarray | None = None,
    ) -> None:
        """
        Initialize the model
        
        Args:
            weights (np.ndarray | None): The weights of the model
            hessian (np.ndarray | None): The Hessian matrix of the model at optimum
        
        Returns:
            None
        """
        if hessian is not None and weights is None:
            raise ValueError("Cannot pass Hessian without providing the corresponding weights.")

        if weights is not None:
            if weights.ndim != 1:
                raise ValueError("weights must be a 1D array.")
            
        if hessian is not None:
            if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
                raise ValueError("Hessian must be a square 2D matrix.")
            
            if hessian.shape[0] != weights.shape[0]:
                raise ValueError(
                    f"Dimension mismatch: Hessian shape {hessian.shape} "
                    f"does not match Weights length ({weights.shape[0]})."
                )

        self._weights = weights
        self._hessian = hessian

    @property
    def weights(self) -> np.ndarray | None:
        """
        Get the weights of the model

        Returns:
            np.ndarray | None: The weights
        """
        return self._weights.copy() if self._weights is not None else None
    
    def _design_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Create the design matrix
        
        Args:
            X (np.ndarray): The input data

        Returns:
            np.ndarray: The design matrix
        """

        return np.hstack((np.ones((X.shape[0], 1)), X))
    
    def _sigmoid(self, z: np.ndarray, overflow_limit: float = 500) -> np.ndarray:
        """
        Compute the sigmoid function with taking into account overflow

        Args:
            z (np.ndarray): The input data
            overflow_limit (float): The limit for overflow

        Returns:
            np.ndarray: The output of the sigmoid function
        """
        return 1 / (1 + np.exp(-np.clip(z, -overflow_limit, overflow_limit)))
    
    def _loss(self, weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the loss function

        Args:
            weights (np.ndarray): The weights of the model
            X (np.ndarray): The input data
            y (np.ndarray): The target data

        Returns:
            float: The loss value
        """
        z = X @ weights
        loss = np.maximum(z, 0) - z * y + np.log(1 + np.exp(-np.abs(z)))
        return np.sum(loss)
    
    def _gradient(self, weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        proba = self._sigmoid(X @ weights)
        return X.T @ (proba - y)
    
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        max_iter: int = 10000,
        max_ls: int = 50,
        tol: float = 1e-12
    ) -> LogisticRegression:
        """
        Fit the model to the data

        Args:
            X (pd.DataFrame): The input data
            y (pd.Series): The target data

        Returns:
            LogisticRegression: The fitted model
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        X = self._design_matrix(X)

        initial_weights = np.ones(X.shape[1])
        result = minimize(
            self._loss,
            initial_weights,
            args=(X, y),
            jac=self._gradient,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'maxls': max_ls},
            tol=tol
        )

        if not result.success:
            raise ValueError(f"Optimization failed to converge: {result.message}")
        
        self._weights = result.x
        proba = self._sigmoid(X @ self._weights)
        variance_matrix = proba * (1 - proba)
        self._hessian = X.T @ (variance_matrix[:, np.newaxis] * X)

        return self
    
    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predict the target data using the model and return the probabilities

        Args:
            X (pd.DataFrame | np.ndarray): The input data

        Returns:
            pd.Series: The predicted target data with probabilities
        """
        if self._weights is None:
            raise ValueError("The model has not been fitted yet")
        
        if X.shape[1] != self._weights.shape[0] - 1:
            raise ValueError("The number of features in X does not match the number of weights in the model")
        
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        X = self._design_matrix(X)

        return self._sigmoid(X @ self._weights)
    
    def predict(self, X: pd.DataFrame | np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict the target data using the model

        Args:
            X (pd.DataFrame | np.ndarray): The input data
            threshold (float): The threshold for the prediction

        Returns:
            pd.Series: The predicted target data as 0 (absence) or 1 (presence)
        """
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def forget(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        n_batches: int = 1,
    ) -> LogisticRegression:
        """
        Forget the specified data.

        Args:
            X (pd.DataFrame | np.ndarray): The data to forget
            y (pd.Series | np.ndarray): The target data to forget
            n_batches (int): Number of sequential chunks to split the forget set into

        Returns:
            LogisticRegression: The updated model
        """
        if self._weights is None:
            raise ValueError("The model has not been fitted yet")

        if self._hessian is None:
            raise ValueError("The hessian matrix has not been calculated yet")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        if X.shape[1] != self._weights.shape[0] - 1:
            raise ValueError("The number of features in X does not match the number of weights in the model")

        if X.shape[0] < n_batches:
            raise ValueError("The number of data points in X cannot be less than n_batches")
        
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        X = self._design_matrix(X)

        new_weights = self._weights.copy()
        new_hessian = self._hessian

        for X_batch, y_batch in zip(np.array_split(X, n_batches), np.array_split(y, n_batches)):
            proba_forget = self._sigmoid(X_batch @ new_weights)
            variance_matrix_forget = proba_forget * (1 - proba_forget)

            gradient_forget = X_batch.T @ (proba_forget - y_batch)
            hessian_forget = X_batch.T @ (variance_matrix_forget[:, np.newaxis] * X_batch)

            new_hessian = new_hessian - hessian_forget
            new_weights = new_weights + np.linalg.solve(new_hessian, gradient_forget)

        return LogisticRegression(weights=new_weights, hessian=new_hessian)

    def rebase_weights(self, func: LogisticRegressionWeightsRebasing) -> LogisticRegression:
        if self._weights is None:
            raise ValueError("The model has not been fitted yet")
        if self._hessian is None:
            raise ValueError("The hessian matrix has not been calculated yet")

        weights, hessian = func.rebase(self._weights, self._hessian)
        return LogisticRegression(weights=weights, hessian=hessian)