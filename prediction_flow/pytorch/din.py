"""
Deep Interest Network.
"""

from .nn import Attention
from .interest_net import InterestNet


class DIN(InterestNet):
    """Deep Interest Network.

    Parameters
    ----------
    features : Features

    attention_groups : list of AttentionGroup

    num_classes : int
        Number of classes.

    embedding_size : int
        Size of embedding.

    hidden_layers : list
        Size of hidden layers.
        Example: [96, 32]

    dnn_activation : str
        Activation function of deep layers.
        Example: relu

    final_activation : str
        Activation function of output.

    dropout : float
        Dropout rate.
    """
    def __init__(self, *args, **kwargs):
        super(DIN, self).__init__(*args, **kwargs)

    def create_attention_fn(self, attention_group):
        return Attention(
            attention_group.pairs_count * self.embedding_size,
            hidden_layers=attention_group.hidden_layers,
            dropout=attention_group.att_dropout,
            activation=attention_group.activation)
