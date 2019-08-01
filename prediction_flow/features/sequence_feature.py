"""
Class for sequence features
"""

# Authors: Hongwei Zhang
# License: MIT


from .base import BaseFeature


class Sequence(BaseFeature):
    """
    Class for sequence features

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

    column_flow : ColumnFlow
        ColumnFlow to transform this feature.
    """
    def __init__(self, name, column_flow):
        super().__init__(name=name, column_flow=column_flow)

    def dimension(self):
        """The dimension of sequence feature is the dimension
        of last transformer in ColumnFlow.
        """
        return self.column_flow.transformers[-1].dimension()

    def max_length(self):
        """The max length of sequence feature is the max length
        of last transformer in ColumnFlow.
        """
        return self.column_flow.transformers[-1].max_length()
