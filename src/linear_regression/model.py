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
        weights = np.asarray(weights) if weights is not None else None
        gram = np.asarray(gram) if gram is not None else None
        moment = np.asarray(moment) if moment is not None else None

        if weights is not None and gram is None and moment is None:
            if weights.ndim != 1:
                raise ValueError("Weights must be a 1D array.")

        elif weights is None and gram is not None and moment is not None:
            gram, moment = self._validate_gram_and_moment(gram, moment)
            try:
                weights = np.linalg.solve(gram, moment)
            except np.linalg.LinAlgError:
                raise ValueError("The provided Gram matrix is singular and cannot be inverted to find weights.")

        elif weights is not None and gram is not None and moment is not None:
            if weights.ndim != 1:
                raise ValueError("Weights must be a 1D array.")
                
            gram, moment = self._validate_gram_and_moment(gram, moment)
            
            if weights.shape[0] != gram.shape[0]:
                raise ValueError(
                    f"Shape mismatch: weights shape {weights.shape} does not match Gram/Moment size ({gram.shape[0]})."
                )
                
            if not np.allclose(gram @ weights, moment, rtol=1e-4, atol=1e-4):
                raise ValueError("Provided weights, gram, and moment are mathematically inconsistent.")

        elif weights is None and gram is None and moment is None:
            pass

        else:
            raise ValueError(
                "Invalid initialization combination. You must provide either:\n"
                "1. Nothing\n"
                "2. Weights only\n"
                "3. Gram and Moment only\n"
                "4. Weights, Gram, AND Moment"
            )

        self._weights = weights
        self._gram = gram
        self._moment = moment

    def _validate_gram_and_moment(self, gram: np.ndarray, moment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Validate the Gram matrix and moment
        
        Args:
            gram (np.ndarray): The gram matrix
            moment (np.ndarray): The moment

        Returns:
            tuple[np.ndarray, np.ndarray]: The validated Gram matrix and moment
        """
        if moment.ndim != 1:
            raise ValueError("Moment must be a 1D array.")
            
        num_features = moment.shape[0]
        
        if gram.shape != (num_features, num_features):
            raise ValueError(
                f"Shape mismatch: Gram matrix shape {gram.shape} must be square and match moment size ({num_features}, {num_features})."
            )
        if not np.allclose(gram, gram.T, atol=1e-8):
            raise ValueError("The Gram matrix must be symmetric.")
        if np.any(np.linalg.eigvalsh(gram) < -1e-8):
            raise ValueError("The Gram matrix must be positive semi-definite.")
            
        return gram, moment

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


