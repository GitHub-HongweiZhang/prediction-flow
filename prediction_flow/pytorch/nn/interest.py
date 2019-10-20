"""
Iinterest part used by DIEN model.

Reference:
    Deep Interest Evolution Network for Click-Through Rate Prediction
    https://arxiv.org/pdf/1809.03672.pdf
"""

# Authors: Hongwei Zhang
# License: MIT


from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .attention import Attention
from .rnn import DynamicGRU


class AuxiliaryNet(nn.Module):
    """ NN for Auxiliary Loss.

    Parameters
    ----------
    input_size : int
        Size of input.

    hidden_layers : iterable
        Hidden layer sizes.

    activation : str
        Name of activation function. ReLU, PReLU and Sigmoid are supported.
    """
    def __init__(self, input_size, hidden_layers, activation='sigmoid'):
        super(AuxiliaryNet, self).__init__()
        modules = OrderedDict()

        previous_size = input_size
        for index, hidden_layer in enumerate(hidden_layers):
            modules[f"dense{index}"] = nn.Linear(previous_size, hidden_layer)
            if activation:
                if activation.lower() == 'relu':
                    modules[f"activation{index}"] = nn.ReLU()
                elif activation.lower() == 'prelu':
                    modules[f"activation{index}"] = nn.PReLU()
                elif activation.lower() == 'sigmoid':
                    modules[f"activation{index}"] = nn.Sigmoid()
                else:
                    raise NotImplementedError(f"{activation} is not supported")
            previous_size = hidden_layer
        modules["final_layer"] = nn.Linear(previous_size, 1)
        self._sequential = nn.Sequential(modules)

    def forward(self, input):
        return torch.sigmoid(self._sequential(input))


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

    use_negsampling : bool
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
            att_activation='prelu',
            use_negsampling=False):
        super(Interest, self).__init__()
        if gru_type not in Interest.__SUPPORTED_GRU_TYPE__:
            raise NotImplementedError(f"gru_type: {gru_type} is not supported")

        self.gru_type = gru_type
        self.use_negsampling = use_negsampling

        self.interest_extractor = nn.GRU(
            input_size=input_size,
            hidden_size=input_size,
            batch_first=True,
            bidirectional=False)

        if self.use_negsampling:
            self.auxiliary_net = AuxiliaryNet(
                input_size * 2, hidden_layers=[100, 50])

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

    def cal_auxiliary_loss(
            self, states, click_seq, noclick_seq, keys_length):
        # states [B, T, H]
        # click_seq [B, T, H]
        # noclick_seq [B, T, H]
        # keys_length [B]
        batch_size, max_seq_length, embedding_size = states.size()

        mask = (torch.arange(max_seq_length, device=states.device).repeat(
            batch_size, 1) < keys_length.view(-1, 1)).float()

        click_input = torch.cat([states, click_seq], dim=-1)
        noclick_input = torch.cat([states, noclick_seq], dim=-1)
        embedding_size = embedding_size * 2

        click_p = self.auxiliary_net(
            click_input.view(
                batch_size * max_seq_length, embedding_size)).view(
                    batch_size, max_seq_length)[mask > 0].view(-1, 1)
        click_target = torch.ones(
            click_p.size(), dtype=torch.float, device=click_p.device)

        noclick_p = self.auxiliary_net(
            noclick_input.view(
                batch_size * max_seq_length, embedding_size)).view(
                    batch_size, max_seq_length)[mask > 0].view(-1, 1)
        noclick_target = torch.zeros(
            noclick_p.size(), dtype=torch.float, device=noclick_p.device)

        loss = F.binary_cross_entropy(
            torch.cat([click_p, noclick_p], dim=0),
            torch.cat([click_target, noclick_target], dim=0))

        return loss

    def forward(self, query, keys, keys_length, neg_keys=None):
        """
        Parameters
        ----------
        query: 2D tensor, [B, H]
        kerys: 3D tensor, [B, T, H]
        keys_length: 1D tensor, [B]
        neg_keys: 3D tensor, [B, T, H]

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

        aloss = None
        if (self.gru_type != 'GRU') or self.use_negsampling:
            interests, _ = pad_packed_sequence(
                packed_interests,
                batch_first=True,
                padding_value=0.0,
                total_length=max_length)

            if self.use_negsampling:
                aloss = self.cal_auxiliary_loss(
                    interests[:, :-1, :],
                    keys[:, 1:, :],
                    neg_keys[:, 1:, :],
                    keys_length - 1)

        if self.gru_type == 'GRU':
            packed_interests, _ = self.interest_evolution(packed_interests)

            interests, _ = pad_packed_sequence(
                packed_interests,
                batch_first=True,
                padding_value=0.0,
                total_length=max_length)

            outputs = self.attention(query, interests, keys_length)

        elif self.gru_type == 'AIGRU':
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

        return outputs, aloss
