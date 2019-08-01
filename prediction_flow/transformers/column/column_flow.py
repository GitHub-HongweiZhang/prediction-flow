"""ColumnFlow contaions a chain of column-orientation
transformers (implementing fit/transform).
"""

# Authors: Hongwei Zhang
# License: MIT


import numpy as np


class ColumnFlow(object):
    """ColumnFlow contaions a chain of column-orientation
    transformers (implementing fit/transform).

    Parameters
    ----------
    transformers : list
        List of column transformers (implementing fit/transform) that are
        chained, in the order in which they are chained.

    verbose : boolean, optional
        If True, the log while fitting each transformer will be printed.

    Attributes
    ----------
    transformers : list
        List of column transformers (implementing fit/transform) that are
        chained, in the order in which they are chained.

    verbose : boolean, optional
        If True, the log while fitting each transformer will be printed.
    """

    def __init__(self, transformers, verbose=False):
        ColumnFlow.__check_transformers(transformers)
        self.transformers = transformers
        self.verbose = verbose

    @staticmethod
    def __check_transformers(transformers):
        if not isinstance(transformers, list):
            raise TypeError(
                "transformers must be list type, not {type(transformers)}")

        types = [
            transformer.column_type for transformer in transformers]

        if len(set(types)) != 1:
            raise ValueError("transformers must be the same type, not {types}")

    def fit(self, x, y=None):
        """Fit all transformers one after the other.

        Parameters
        ----------
        x : array-like
            One column of training data.
        y : array-like, default=None
            Training targets.

        Returns
        -------
        self : ColumnFlow
            This flow.
        """
        transformed_x = np.asarray(x).ravel()
        for transformer in self.transformers:
            transformer.fit(transformed_x, y)
            transformed_x = transformer.transform(transformed_x)

        return self

    def transform(self, x):
        """Transform x by all fitted transformers.

        Parameters
        ----------
        x : array-like
            Column data to be transformed.

        Returns
        -------
        transformed_x : array-like
            Transformed data.
        """
        transformed_x = np.asarray(x).ravel()
        for transformer in self.transformers:
            transformed_x = transformer.transform(transformed_x)

        return transformed_x
