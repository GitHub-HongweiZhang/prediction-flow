"""
StandardScaler to convert term to number.
"""

# Authors: Hongwei Zhang
# License: MIT


import numpy as np
import sklearn.preprocessing as sk

from .base import NumberColumn


class StandardScaler(NumberColumn):
    """Normalize number feature.
    """
    def __init__(self):
        self.__scaler = sk.StandardScaler()

    def fit(self, x, y=None):
        """Fit this transformer.

        Parameters
        ----------
        x : array-like
            One column of training data.
        y : array-like, default=None, ignored
            Training targets.

        Returns
        -------
        self : StandardScaler
            This StandardScaler.
        """
        self.__scaler.fit(np.asarray(x, dtype=np.float).reshape(-1, 1))
        return self

    def transform(self, x):
        """Transform x by this fitted transformer.

        Parameters
        ----------
        x : array-like
            Column data to be transformed.

        Returns
        -------
        transformed_x : array-like
            Transformed data.
        """
        transformed_x = self.__scaler.transform(
            np.asarray(x, dtype=np.float32).reshape(-1, 1))

        return transformed_x.ravel()
