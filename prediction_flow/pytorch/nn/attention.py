"""
Attention used by DIN model.

Reference:
    Deep Interest Network for Click-Through Rate Prediction
    https://github.com/zhougr1993/DeepInterestNetwork/blob/master/din/model.py
"""

# Authors: Hongwei Zhang
# License: MIT


import numpy as np

import torch
import torch.nn as nn
from torch.functional import F

from .mlp import MLP


class Attention(nn.Module):
    """Attention layer.

    Parameters
    ----------
    input_size : int
        Size of input.

    hidden_layers : iterable
        Hidden layer sizes.

    dropout : float
        Dropout rate.

    activation : str
        Name of activation function. relu, prelu and sigmoid are supported.

    return_scores : bool
        Return attention scores instead of weighted sum pooling result.
    """
    def __init__(
            self,
            input_size,
            hidden_layers,
            dropout=0.0,
            batchnorm=True,
            activation='prelu',
            return_scores=False):
        super(Attention, self).__init__()
        self.return_scores = return_scores
        self.mlp = MLP(
            input_size=input_size * 4,
            hidden_layers=hidden_layers,
            dropout=dropout,
            batchnorm=batchnorm,
            activation=activation)
        self.fc = nn.Linear(hidden_layers[-1], 1)

    def forward(self, query, keys, keys_length):
        """
        Parameters
        ----------
        query: 2D tensor, [B, H]
        kerys: 3D tensor, [B, T, H]
        keys_length: 1D tensor, [B]

        Returns
        -------
        outputs: 2D tensor, if return_scores=False [B, H], otherwise [B, T]
        """
        batch_size, max_length, dim = keys.size()

        query = query.unsqueeze(1).expand(-1, max_length, -1)

        din_all = torch.cat(
            [query, keys, query - keys, query * keys], dim=-1)

        din_all = din_all.view(batch_size * max_length, -1)

        outputs = self.mlp(din_all)

        outputs = self.fc(outputs).view(batch_size, max_length)  # [B, T]

        # Scale
        outputs = outputs / (dim ** 0.5)

        # Mask
        mask = (torch.arange(max_length, device=keys_length.device).repeat(
            batch_size, 1) < keys_length.view(-1, 1))
        outputs[~mask] = -np.inf

        # Activation
        outputs = F.softmax(outputs, dim=1)  # [B, T]

        if not self.return_scores:
            # Weighted sum
            outputs = torch.matmul(
                outputs.unsqueeze(1), keys).squeeze()  # [B, H]

        return outputs
