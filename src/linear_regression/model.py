from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.linear_regression.weights_rebasing import LinearRegressionWeightsRebasing

class LinearRegression():

    def __init__(
        self,
        weights: np.ndarray | None = None,
        gram: np.ndarray | None = None,
        moment: np.ndarray | None = None
    ) -> None:
        """
        Initialize the model
        
        Args:
            weights (np.ndarray | None): The weights of the model
            gram (np.ndarray | None): The gram matrix of the model
            moment (np.ndarray | None): The moment of the model
        
        Returns:
            None
        """

        self._weights, self._gram, self._moment = weights, gram, moment

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

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> LinearRegression:
        """
        Fit the model to the data

        Args:
            X (pd.DataFrame): The input data
            y (pd.Series): The target data

        Returns:
            LinearRegression: The fitted model
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
        
        self._gram = X.T @ X
        self._moment = X.T @ y

        try:
            self._weights = np.linalg.solve(self._gram, self._moment)
        except np.linalg.LinAlgError:
            raise ValueError("Forgetting this data makes the Gram matrix singular (not enough remaining variance).")

        return self
    
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predict the target data using the model

        Args:
            X (pd.DataFrame | np.ndarray): The input data

        Returns:
            pd.Series: The predicted target data
        """
        if self._weights is None:
            raise ValueError("The model has not been fitted yet")
        
        if X.shape[1] != self._weights.shape[0] - 1:
            raise ValueError("The number of features in X does not match the number of weights in the model")
        
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        X = self._design_matrix(X)

        return X @ self._weights
    
    def r2_score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Calculate r2 score
        
        Args:
            X (pd.DataFrame): The input data
            y (pd.Series): The target data

        Returns:
            float: The r2 score
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        
        y = np.asarray(y.values)
        y_pred = self.predict(X)
        ss_res = np.power(y - y_pred, 2).sum()
        ss_tot = np.power(y - y.mean(), 2).sum()
        return 1 - ss_res / ss_tot
    
    def mse_score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Calculate mse score
        
        Args:
            X (pd.DataFrame): The input data
            y (pd.Series): The target data

        Returns:
            float: The mse score
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        
        y = np.asarray(y.values)
        y_pred = self.predict(X)
        return np.power(y - y_pred, 2).mean()
    
    def forget(self, X: pd.DataFrame, y: pd.Series) -> LinearRegression:
        """
        Forget the specified data used to fit the model

        Args:
            X (pd.DataFrame): The data to forget
            y (pd.Series): The target data to forget

        Returns:
            LinearRegression: The model
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        
        if self._weights is None:
            raise ValueError("The model has not been fitted yet")
        
        if self._gram is None:
            raise ValueError("The gram matrix has not been calculated yet")
        
        if self._moment is None:
            raise ValueError("The moment has not been calculated yet")
        
        if X.shape[1] != self._weights.shape[0] - 1:
            raise ValueError("The number of features in X does not match the number of weights in the model")
        
        X, y = np.asarray(X.values), np.asarray(y.values)
        X = self._design_matrix(X)

        gram = self._gram - X.T @ X
        moment = self._moment - X.T @ y

        try:
            weights = np.linalg.solve(gram, moment)
        except np.linalg.LinAlgError:
            raise ValueError("Forgetting this data makes the Gram matrix singular (not enough remaining variance).")
        
        return LinearRegression(weights=weights, gram=gram, moment=moment)
    
    def rebase_weights(
        self, 
        func: LinearRegressionWeightsRebasing
    ) -> LinearRegression:
        """
        Rebase the weights of the model

        Args:
            func (Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]): The rebase function

        Returns:
            LinearRegression: The model
        """
        if self._weights is None:
            raise ValueError("The model has not been fitted yet")
        
        if self._gram is None:
            raise ValueError("The gram matrix has not been calculated yet")
        
        if self._moment is None:
            raise ValueError("The moment has not been calculated yet")
        
        weights, gram, moment = func.rebase(self._weights, self._gram, self._moment)
        return LinearRegression(weights=weights, gram=gram, moment=moment)


