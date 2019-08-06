"""
Multilayer perceptron torch module.
"""

# Authors: Hongwei Zhang
# License: MIT


from collections import OrderedDict

import torch.nn as nn


class MLP(nn.Module):
    """Multilayer perceptron torch module.

    Parameters
    ----------
    input_size : int
        Size of input.

    hidden_layers : iterable
        Hidden layer sizes.

    dropout : float
        Dropout rate.

    activation : str
        Name of activation function. ReLU, PReLU and Sigmoid are supported.
    """
    def __init__(self, input_size, hidden_layers,
                 dropout=0.0, batchnorm=True, activation='relu'):
        super(MLP, self).__init__()
        modules = OrderedDict()

        previous_size = input_size
        for index, hidden_layer in enumerate(hidden_layers):
            modules[f"dense{index}"] = nn.Linear(previous_size, hidden_layer)
            if batchnorm:
                modules[f"batchnorm{index}"] = nn.BatchNorm1d(hidden_layer)
            if activation:
                if activation.lower() == 'relu':
                    modules[f"activation{index}"] = nn.ReLU()
                elif activation.lower() == 'prelu':
                    modules[f"activation{index}"] = nn.PReLU()
                elif activation.lower() == 'sigmoid':
                    modules[f"activation{index}"] = nn.Sigmoid()
                else:
                    raise NotImplementedError(f"{activation} is not supported")
            if dropout:
                modules[f"dropout{index}"] = nn.Dropout(dropout)
            previous_size = hidden_layer
        self._sequential = nn.Sequential(modules)

    def forward(self, input):
        return self._sequential(input)
