"""AttentionGRU and AttentionUpdateGateGRU.
"""

# Authors: Hongwei Zhang
# License: MIT


import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence


class AttentionGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(AttentionGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # (W_ir|W_iz|W_in)
        self.weight_ih = nn.Parameter(
            torch.Tensor(3 * hidden_size, input_size))
        # (W_hr|W_hz|W_hn)
        self.weight_hh = nn.Parameter(
            torch.Tensor(3 * hidden_size, hidden_size))
        if bias:
            # (b_ir|b_iz|b_in)
            self.bias_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
            # (b_hr|b_hz|b_hn)
            self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, input, hx, att_score):
        """

        References
        ----------
            https://github.com/pytorch/pytorch/blob/v0.4.1/torch/nn/_functions/rnn.py#L49
        """

        gi = F.linear(input, self.weight_ih, self.bias_ih)
        gh = F.linear(hx, self.weight_hh, self.bias_hh)
        i_r, i_z, i_n = gi.chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        # updategate = torch.sigmoid(i_z + h_z)
        newgate = torch.tanh(i_n + resetgate * h_n)
        # hy = newgate + updategate * (hx - newgate)

        att_score = att_score.view(-1, 1)

        hy = (1. - att_score) * hx + att_score * newgate

        return hy


class AttentionUpdateGateGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(AttentionUpdateGateGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # (W_ir|W_iz|W_in)
        self.weight_ih = nn.Parameter(
            torch.Tensor(3 * hidden_size, input_size))
        # (W_hr|W_hz|W_hn)
        self.weight_hh = nn.Parameter(
            torch.Tensor(3 * hidden_size, hidden_size))
        if bias:
            # (b_ir|b_iz|b_in)
            self.bias_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
            # (b_hr|b_hz|b_hn)
            self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, input, hx, att_score):
        """

        References
        ----------
            https://github.com/pytorch/pytorch/blob/v0.4.1/torch/nn/_functions/rnn.py#L49
        """

        gi = F.linear(input, self.weight_ih, self.bias_ih)
        gh = F.linear(hx, self.weight_hh, self.bias_hh)
        i_r, i_z, i_n = gi.chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        updategate = torch.sigmoid(i_z + h_z)
        newgate = torch.tanh(i_n + resetgate * h_n)

        updategate = att_score.view(-1, 1) * updategate

        hy = newgate + updategate * (hx - newgate)

        return hy


class DynamicGRU(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, gru_type='AGRU'):
        super(DynamicGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        if gru_type == 'AGRU':
            self.rnn = AttentionGRUCell(input_size, hidden_size, bias)
        elif gru_type == 'AUGRU':
            self.rnn = AttentionUpdateGateGRUCell(
                input_size, hidden_size, bias)

    def forward(self, input, att_scores, hx=None):
        is_packed_input = isinstance(input, PackedSequence)
        if not is_packed_input:
            raise NotImplementedError(
                "DynamicGRU only supports packed input")

        is_packed_att_scores = isinstance(att_scores, PackedSequence)
        if not is_packed_att_scores:
            raise NotImplementedError(
                "DynamicGRU only supports packed att_scores")

        input, batch_sizes, sorted_indices, unsorted_indices = input
        att_scores, _, _, _ = att_scores

        max_batch_size = batch_sizes[0]
        max_batch_size = int(max_batch_size)

        if hx is None:
            hx = torch.zeros(
                max_batch_size, self.hidden_size,
                dtype=input.dtype, device=input.device)

        outputs = torch.zeros(
            input.size(0), self.hidden_size,
            dtype=input.dtype, device=input.device)

        begin = 0
        for batch in batch_sizes:
            new_hx = self.rnn(
                input[begin: begin + batch],
                hx[0:batch],
                att_scores[begin: begin + batch])
            outputs[begin: begin + batch] = new_hx
            hx = new_hx
            begin += batch

        return PackedSequence(
            outputs, batch_sizes, sorted_indices, unsorted_indices)
