from abc import ABC, abstractmethod
import numpy as np
from src.scalers.standard_scaler import StandardScaler


class LogisticRegressionWeightsRebasing(ABC):
    @abstractmethod
    def rebase(
        self,
        weights: np.ndarray,
        hessian: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Rebase the weights of the model. 

        Args:
            weights (np.ndarray): The weights of the model
            hessian (np.ndarray): The hessian of the model

        Returns:
            tuple[np.ndarray, np.ndarray]: The weights, the hessian
        """
        raise NotImplementedError


class LogisticRegressionStandardScaleWeightsRebasing(LogisticRegressionWeightsRebasing):
    def __init__(
        self,
        source_scaler: StandardScaler,
        target_scaler: StandardScaler
    ) -> None:
        """
        Initialize the LinearRegressionStandardScaleWeightsRebasing class
        
        Args:
            source_scaler (StandardScaler): The source scaler
            target_scaler (StandardScaler): The destination scaler
        
        Returns:
            None
        """
        self._source_scaler = source_scaler
        self._target_scaler = target_scaler

    def rebase(
        self,
        weights: np.ndarray,
        hessian: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Rebase the weights of the model.
        
        Args:
            weights (np.ndarray): The weights of the model
            hessian (np.ndarray): The hessian of the model

        Returns:
            tuple[np.ndarray, np.ndarray]: The weights, the hessian
        """
        source_mean, source_std = self._source_scaler.mean.values, self._source_scaler.std.values
        target_mean, target_std = self._target_scaler.mean.values, self._target_scaler.std.values

        n_features = source_mean.shape[0]
        change_basis_matrix = np.eye(n_features + 1)
        change_basis_matrix[0, 1:] = (source_mean - target_mean) / target_std
        change_basis_matrix[1:, 1:] = np.diag(source_std / target_std)

        new_weights = np.linalg.solve(change_basis_matrix, weights)
        new_hessian = change_basis_matrix.T @ hessian @ change_basis_matrix
        return new_weights, new_hessian