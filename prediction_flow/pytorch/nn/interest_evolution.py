"""
Iinterest evolution part used by DIEN model.

Reference:
    Deep Interest Evolution Network for Click-Through Rate Prediction
    https://arxiv.org/pdf/1809.03672.pdf
"""

# Authors: Hongwei Zhang
# License: MIT


import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .din_attention import Attention


class InterestEvolution(nn.Module):
    """InterestEvolution layer.

    Parameters
    ----------
    input_size : int
        Size of input.

    gru_type : str
        Type of GRU. GRU, AIGRU, AGRU and AUGRU are supported.

    gru_dropout : float
        Dropout rate of GRU.

    att_hidden_layers : iterable
        Hidden layer sizes of attention.

    att_dropout : float
        Dropout rate of attention.

    att_batchnorm : bool
        Batchnorm of attention.

    att_activation : str
        Activation function name of attention.
        relu, prelu and sigmoid are supported.
    """
    __SUPPORTED_GRU_TYPE__ = ['GRU']

    def __init__(
            self,
            input_size,
            gru_type='GRU',
            gru_dropout=0.0,
            att_hidden_layers=[80, 40],
            att_dropout=0.0,
            att_batchnorm=True,
            att_activation='prelu'):
        super(InterestEvolution, self).__init__()
        if gru_type not in InterestEvolution.__SUPPORTED_GRU_TYPE__:
            raise NotImplementedError(f"gru_type: {gru_type} is not supported")

        self.gru_type = gru_type

        self.interest_extractor = nn.GRU(
            input_size=input_size,
            hidden_size=input_size,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=False)

        if gru_type == 'GRU':
            self.interest_evolution = nn.GRU(
                input_size=input_size,
                hidden_size=input_size,
                batch_first=True,
                dropout=gru_dropout,
                bidirectional=False)

            self.attention = Attention(
                input_size=input_size,
                hidden_layers=att_hidden_layers,
                dropout=att_dropout,
                batchnorm=att_batchnorm,
                activation=att_activation)

    def forward(self, query, keys, keys_length):
        """
        Parameters
        ----------
        query: 2D tensor, [B, H]
        kerys: 3D tensor, [B, T, H]
        keys_length: 1D tensor, [B]

        Returns
        -------
        outputs: 2D tensor, [B, H]
        """
        batch_size, max_length, dim = keys.size()

        packed_keys = pack_padded_sequence(
            keys,
            lengths=keys_length.squeeze(),
            batch_first=True,
            enforce_sorted=False)

        packed_interests, _ = self.interest_extractor(packed_keys)

        if self.gru_type == 'GRU':
            packed_interests, _ = self.interest_evolution(packed_interests)
            interests, _ = pad_packed_sequence(
                packed_interests,
                batch_first=True,
                padding_value=0.0,
                total_length=max_length)

            outputs = self.attention(query, interests, keys_length)

        return outputs
