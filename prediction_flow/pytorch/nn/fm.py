"""
FM layer.
"""

# Authors: Hongwei Zhang
# License: MIT


import torch
import torch.nn as nn


class FM(nn.Module):
    """FM layer.
    """
    def __init__(self, reduce_sum=True):
        super(FM, self).__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        sum_squared = torch.pow(torch.sum(x, dim=1), 2)
        squared_sum = torch.sum(torch.pow(x, 2), dim=1)
        second_order = sum_squared - squared_sum
        if self.reduce_sum:
            output = torch.sum(second_order, dim=1, keepdim=True)
        return 0.5 * output
