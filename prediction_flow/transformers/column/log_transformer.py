"""
LogTransformer to convert number feature.
"""

# Authors: Hongwei Zhang
# License: MIT


import numpy as np

from .base import NumberColumn


class LogTransformer(NumberColumn):
    """LogTransformer to convert number feature.
    """
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
        self : LogTransformer
            This LogTransformer.
        """
        return self

    def transform(self, x):
        """ log(1 + x) when x > 0 else x

        Parameters
        ----------
        x : array-like
            Column data to be transformed.

        Returns
        ----------
        res: array-like
        """
        res = x.copy().astype(np.float).ravel()
        mask = x > 0.0
        res[mask] = np.log(1 + x[mask])
        return res
