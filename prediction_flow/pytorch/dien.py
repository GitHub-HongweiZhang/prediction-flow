"""
Deep Interest Evolution Network.
"""

from .nn import InterestEvolution
from .interest_net import InterestNet


class DIEN(InterestNet):
    """Deep Interest Evolution Network.

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
        super(DIEN, self).__init__(*args, **kwargs)

    def create_attention_fn(self, attention_group):
        return InterestEvolution(
            attention_group.pairs_count * self.embedding_size,
            gru_type=attention_group.gru_type,
            gru_dropout=attention_group.gru_dropout,
            att_hidden_layers=attention_group.hidden_layers,
            att_dropout=attention_group.att_dropout,
            att_activation=attention_group.activation)
