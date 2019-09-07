"""
Deep Neural Network.
"""
from collections import OrderedDict

import torch
import torch.nn as nn

from .base import EmbeddingMixin
from .nn import MLP, MaxPooling
from .utils import init_weights


class DNN(nn.Module, EmbeddingMixin):
    """Deep Neural Network.

    Parameters
    ----------
    features : Features

    num_classes : int
        Number of classes.

    embedding_size : int
        Size of embedding.

    hidden_layers : list
        Size of hidden layers.
        Example: [96, 32]

    activation : str
        Activation function.
        Example: relu

    final_activation : str
        Activation function of output.

    dropout : float
        Dropout rate.
    """
    def __init__(self, features, num_classes, embedding_size, hidden_layers,
                 activation='relu', final_activation=None, dropout=0.0):
        super(DNN, self).__init__()
        self.features = features
        self.num_classes = num_classes
        self.final_activation = final_activation

        self.embeddings, self.embedding_sizes = self.build_embeddings(
            embedding_size)

        self._sequence_poolings = OrderedDict()

        total_embedding_sizes = 0
        for feature in self.features.category_features:
            total_embedding_sizes += (
                self.embedding_sizes[feature.name])

        for feature in self.features.sequence_features:
            self._sequence_poolings[feature.name] = MaxPooling(1)
            self.add_module(
                f"pooling:{feature.name}",
                self._sequence_poolings[feature.name])
            total_embedding_sizes += (
                self.embedding_sizes[feature.name])

        total_input_size = (total_embedding_sizes +
                            len(self.features.number_features))
        self.mlp = MLP(
            total_input_size,
            hidden_layers,
            dropout=dropout, batchnorm=True, activation=activation)
        final_layer_input_size = hidden_layers[-1]

        output_size = self.num_classes

        if self.num_classes == 2 and self.final_activation == 'sigmoid':
            output_size -= 1

        self.final_layer = nn.Linear(final_layer_input_size, output_size)

        self.apply(init_weights)

    def forward(self, x):
        final_layer_inputs = list()

        # linear
        number_inputs = list()
        for feature in self.features.number_features:
            number_inputs.append(x[feature.name].view(-1, 1))

        embeddings = list()
        for feature in self.features.category_features:
            embeddings.append(
                self.embeddings[feature.name](x[feature.name]))

        for feature in self.features.sequence_features:
            embeddings.append(
                self._sequence_poolings[feature.name](
                    self.embeddings[feature.name](x[feature.name])))

        emb_concat = torch.cat(number_inputs + embeddings, dim=1)

        final_layer_inputs = self.mlp(emb_concat)

        output = self.final_layer(final_layer_inputs)

        if self.num_classes == 2 and self.final_activation == 'sigmoid':
            output = torch.sigmoid(output)
        elif self.num_classes > 1 and self.final_activation == 'softmax':
            output = torch.softmax(output)
        elif self.final_activation:
            raise NotImplementedError(
                f"pair (final_activation: {self.final_activation}, "
                f"num_classes: {self.num_classes}) is not implemented")

        return output
