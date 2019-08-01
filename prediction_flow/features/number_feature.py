"""
Class for number feature.
"""

# Authors: Hongwei Zhang
# License: MIT


from .base import BaseFeature


class Number(BaseFeature):
    """
    Class for number feature.

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
