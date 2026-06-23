import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from abc import ABC, abstractmethod

import numpy as np

from src.scalers.standard_scaler import StandardScaler

class LinearRegressionWeightsRebasing(ABC):
    @abstractmethod
    def rebase(
        self,
        weights: np.ndarray,
        gram: np.ndarray,
        moment: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Rebase the weights of the model. 

        Args:
            weights (np.ndarray): The weights of the model
            gram (np.ndarray): The gram matrix of the model
            moment (np.ndarray): The moment of the model

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: The weights, the gram matrix, and the moment respectively
        """
        raise NotImplementedError
    
class LinearRegressionStandardScaleWeightsRebasing(LinearRegressionWeightsRebasing):
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
        gram: np.ndarray,
        moment: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Rebase the weights of the model.
        
        Args:
            weights (np.ndarray): The weights of the model
            gram (np.ndarray): The gram matrix of the model
            moment (np.ndarray): The moment of the model

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: The weights, the gram matrix, and the moment respectively
        """
        source_mean, source_std = self._source_scaler.mean.values, self._source_scaler.std.values
        target_mean, target_std = self._target_scaler.mean.values, self._target_scaler.std.values

        n_features = source_mean.shape[0]
        change_basis_matrix = np.eye(n_features + 1)

        change_basis_matrix[0, 1:] = (source_mean - target_mean) / target_std
        change_basis_matrix[1:, 1:] = np.diag(source_std / target_std)

        gram = change_basis_matrix.T @ gram @ change_basis_matrix
        moment = change_basis_matrix.T @ moment
        weights = np.linalg.solve(gram, moment)
        return weights, gram, moment