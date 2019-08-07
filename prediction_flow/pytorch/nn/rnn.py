"""
"""

# Authors: Hongwei Zhang
# License: MIT


import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


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
        hy = (1. - att_score) * hx + att_score * newgate

        return hy
