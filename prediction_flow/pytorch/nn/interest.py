"""
Iinterest part used by DIEN model.

Reference:
    Deep Interest Evolution Network for Click-Through Rate Prediction
    https://arxiv.org/pdf/1809.03672.pdf
"""

# Authors: Hongwei Zhang
# License: MIT


import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .attention import Attention
from .rnn import DynamicGRU


class Interest(nn.Module):
    """Interest layer.

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
    __SUPPORTED_GRU_TYPE__ = ['GRU', 'AIGRU', 'AGRU', 'AUGRU']

    def __init__(
            self,
            input_size,
            gru_type='GRU',
            gru_dropout=0.0,
            att_hidden_layers=[80, 40],
            att_dropout=0.0,
            att_batchnorm=True,
            att_activation='prelu'):
        super(Interest, self).__init__()
        if gru_type not in Interest.__SUPPORTED_GRU_TYPE__:
            raise NotImplementedError(f"gru_type: {gru_type} is not supported")

        self.gru_type = gru_type

        self.interest_extractor = nn.GRU(
            input_size=input_size,
            hidden_size=input_size,
            batch_first=True,
            bidirectional=False)

        if gru_type == 'GRU':
            self.interest_evolution = nn.GRU(
                input_size=input_size,
                hidden_size=input_size,
                batch_first=True,
                bidirectional=False)

            self.attention = Attention(
                input_size=input_size,
                hidden_layers=att_hidden_layers,
                dropout=att_dropout,
                batchnorm=att_batchnorm,
                activation=att_activation)
        elif gru_type == 'AIGRU':
            self.attention = Attention(
                input_size=input_size,
                hidden_layers=att_hidden_layers,
                dropout=att_dropout,
                batchnorm=att_batchnorm,
                activation=att_activation,
                return_scores=True)

            self.interest_evolution = nn.GRU(
                input_size=input_size,
                hidden_size=input_size,
                batch_first=True,
                bidirectional=False)
        elif gru_type == 'AGRU' or gru_type == 'AUGRU':
            self.attention = Attention(
                input_size=input_size,
                hidden_layers=att_hidden_layers,
                dropout=att_dropout,
                batchnorm=att_batchnorm,
                activation=att_activation,
                return_scores=True)

            self.interest_evolution = DynamicGRU(
                input_size=input_size,
                hidden_size=input_size,
                gru_type=gru_type)

    @staticmethod
    def _get_last_state(states, keys_length):
        # states [B, T, H]
        batch_size, max_seq_length, hidden_size = states.size()

        mask = (torch.arange(max_seq_length, device=keys_length.device).repeat(
            batch_size, 1) == (keys_length.view(-1, 1) - 1))

        return states[mask]

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

        elif self.gru_type == 'AIGRU':
            interests, _ = pad_packed_sequence(
                packed_interests,
                batch_first=True,
                padding_value=0.0,
                total_length=max_length)

            # attention
            scores = self.attention(query, interests, keys_length)
            interests = interests * scores.unsqueeze(-1)

            packed_interests = pack_padded_sequence(
                interests,
                lengths=keys_length.squeeze(),
                batch_first=True,
                enforce_sorted=False)
            _, outputs = self.interest_evolution(packed_interests)
            outputs = outputs.squeeze()

        elif self.gru_type == 'AGRU' or self.gru_type == 'AUGRU':
            interests, _ = pad_packed_sequence(
                packed_interests,
                batch_first=True,
                padding_value=0.0,
                total_length=max_length)

            # attention
            scores = self.attention(query, interests, keys_length)

            packed_interests = pack_padded_sequence(
                interests,
                lengths=keys_length.squeeze(),
                batch_first=True,
                enforce_sorted=False)

            packed_scores = pack_padded_sequence(
                scores,
                lengths=keys_length.squeeze(),
                batch_first=True,
                enforce_sorted=False)

            outputs, _ = pad_packed_sequence(
                self.interest_evolution(
                    packed_interests, packed_scores), batch_first=True)
            # pick last state
            outputs = Interest._get_last_state(
                outputs, keys_length.squeeze())

        return outputs
