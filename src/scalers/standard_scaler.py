from __future__ import annotations

import numpy as np
import pandas as pd

class StandardScaler:
    def __init__(self, mean: pd.Series | None = None, std: pd.Series | None = None, length: int | None = None):
        """
        Initialize the StandardScaler
        """
        if mean is not None and std is not None and mean.shape[0] != std.shape[0]:
            raise ValueError("Mean and standard deviation must have the same length")
        
        self._mean = mean
        self._std = std
        self._length = length

    def fit(self, X: pd.DataFrame) -> StandardScaler:
        """
        Fit the StandardScaler to the data

        Args:
            X (pd.DataFrame): The data to fit the StandardScaler to

        Returns:
            StandardScaler: The fitted StandardScaler
        """
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0, ddof=0)
        self._length = X.shape[0]
        return self
    
    def forget(self, X: pd.DataFrame) -> StandardScaler:
        """
        Forget the specified data used to fit the StandardScaler

        Args:
            X (pd.DataFrame): The data to forget
        
        Returns:
            StandardScaler: The StandardScaler
        """

        train_length = self._length - X.shape[0]
        if train_length <= 0:
            raise ValueError("Cannot forget all or more data than was originally fitted.")

        train_sum = self._mean * self._length - X.sum(axis=0)
        train_square_sum = self._length * (self._std.pow(2) + self._mean.pow(2))
        train_square_sum -= X.shape[0] * (X.std(axis=0, ddof=0).pow(2) + X.mean(axis=0).pow(2))

        mean = train_sum / train_length
        std  = (train_square_sum / train_length - mean.pow(2)).clip(lower=0).pow(0.5)
        
        return StandardScaler(mean=mean, std=std, length=train_length)
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using the fitted StandardScaler

        Args:
            X (pd.DataFrame): The data to transform

        Returns:
            pd.DataFrame: The transformed data
        """
        if self._mean is None or self._std is None:
            raise ValueError("The StandardScaler has not been fitted yet")
        
        return (X - self._mean) / self._std
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform the data using the fitted StandardScaler

        Args:
            X (pd.DataFrame): The data to inverse transform

        Returns:
            pd.DataFrame: The inverse transformed data
        """
        if self._mean is None or self._std is None:
            raise ValueError("The StandardScaler has not been fitted yet")
        
        return X * self._std + self._mean
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform the data using the StandardScaler

        Args:
            X (pd.DataFrame): The data to transform

        Returns:
            pd.DataFrame: The transformed data
        """
        return self.fit(X).transform(X)
    
    @property
    def length(self) -> int | None:
        """
        Get the length of the data

        Returns:
            int | None: The length
        """
        return self._length
    
    @property
    def mean(self) -> pd.Series| None:
        """
        Get the mean of the data

        Returns:
            pd.Series | None: The mean
        """
        return self._mean
    
    @property
    def std(self) -> pd.Series | None:
        """
        Get the standard deviation of the data

        Returns:
            pd.Series | None: The standard deviation
        """
        return self._std
    
