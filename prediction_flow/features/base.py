"""
Base class for features.
"""

# Authors: Hongwei Zhang
# License: MIT


from abc import ABC, abstractmethod
from ..transformers.column import Column, ColumnFlow


class BaseFeature(ABC):
    """Base class for all feature classes.

    Parameters
    ----------
    name : str
        Name of this feature.

    column_flow : ColumnFlow
        ColumnFlow to transform this feature.

    Attributes
    ----------
    name : str
        Name of this feature.

    column_flow : ColumnFlow or
                  list of column transformers or
                  single transformer
        ColumnFlow to transform this feature.
    """

    @abstractmethod
    def __init__(self, name, column_flow=None):
        self.name = name

        self.column_flow = None

        if column_flow:
            if isinstance(column_flow, ColumnFlow):
                self.column_flow = column_flow
            elif isinstance(column_flow, list):
                self.column_flow = ColumnFlow(column_flow)
            elif isinstance(column_flow, Column):
                self.column_flow = ColumnFlow([column_flow])
            else:
                raise NotImplementedError(
                    "column_flow should be ColumnFlow or "
                    "list of column transformers or single transformer")
