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
    def __init__(self):
        super(FM, self).__init__()

    def forward(self, x):
        sum_squared = torch.pow(torch.sum(x, dim=1), 2).unsqueeze(1)
        squared_sum = torch.sum(torch.pow(x, 2), dim=1).unsqueeze(1)
        second_order = 0.5 * (sum_squared - squared_sum)
        return second_order
