"""
Pooling layers.
"""

# Authors: Hongwei Zhang
# License: MIT


import torch
import torch.nn as nn


class MaxPooling(nn.Module):
    """Max Pooling.

    Parameters
    ----------
    dim : int
        The dimension to do pooling.

    Attributes
    ----------
    dim : int
        The dimension to do pooling.
    """
    def __init__(self, dim):
        super(MaxPooling, self).__init__()
        self.dim = dim

    def forward(self, input):
        return torch.max(input, self.dim)[0]


class SumPooling(nn.Module):
    """Sum Pooling.

    Parameters
    ----------
    dim : int
        The dimension to do pooling.

    Attributes
    ----------
    dim : int
        The dimension to do pooling.
    """
    def __init__(self, dim):
        super(SumPooling, self).__init__()
        self.dim = dim

    def forward(self, input):
        return torch.sum(input, self.dim)
