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

    embedding_name: str
        Embedding name for reference. Give same embedding name to features that
        share same embedding layer.

    embedding_size: int
        Dimension of embedding layer.

    vocab_size: int
        Provide vocab_size if this feature do not need to be pre-processed.
        vocab_size is only be used when column_flow is None.

    max_length: int
        max_length is only be used when column_flow is None.

    Attributes
    ----------
    name : str
        Name of this feature.

    column_flow : ColumnFlow
        ColumnFlow to transform this feature.

    embedding_name: str
        Embedding name for reference. Give same embedding name to features that
        share same embedding layer.

    embedding_size: int
        Dimension of embedding layer.
    """
    def __init__(self, name, column_flow,
                 embedding_name=None, embedding_size=None,
                 vocab_size=None, max_length=None):
        super().__init__(name=name, column_flow=column_flow)
        self.embedding_name = embedding_name if embedding_name else name
        self.embedding_size = embedding_size
        self._vocab_size = vocab_size
        self._max_length = max_length

    def dimension(self):
        """The dimension (vocab size) of sequence feature is the dimension
        of last transformer in ColumnFlow.
        """
        if self.column_flow is not None:
            return self.column_flow.transformers[-1].dimension()
        else:
            if self._vocab_size:
                return self._vocab_size
            else:
                raise RuntimeError(
                    "If param column_flow is not given, "
                    "vocab_size must be given")

    def max_length(self):
        """The max length of sequence feature is the max length
        of last transformer in ColumnFlow.
        """
        if self.column_flow is not None:
            return self.column_flow.transformers[-1].max_length()
        else:
            if self._max_length:
                return self._max_length
            else:
                raise RuntimeError(
                    "If param column_flow is not given, "
                    "max_length must be given")
