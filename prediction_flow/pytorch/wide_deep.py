"""
Wide&Deep Model.
"""
from collections import OrderedDict

import torch
import torch.nn as nn

from .base import EmbeddingMixin
from .nn import MLP, SumPooling
from .utils import init_weights


class WideDeep(nn.Module, EmbeddingMixin):
    """Wide&Deep Model.

    Parameters
    ----------
    features : Features

    wide_features : list of str
        Feature names for wide part.

    deep_features : list of str
        Feature names for deep part.

    cross_features: list of tuple
        Cross sparse feature names for wide part.

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
    def __init__(self, features, wide_features, deep_features, cross_features,
                 num_classes, embedding_size, hidden_layers,
                 activation='relu', final_activation=None, dropout=0.0):
        super(WideDeep, self).__init__()
        self.features = features
        self.wide_features = wide_features
        self.deep_features = deep_features
        self.cross_features = cross_features
        self.num_classes = num_classes
        self.final_activation = final_activation

        self.embeddings, self.embedding_sizes = self.build_embeddings(
            embedding_size)

        self._sequence_poolings = OrderedDict()

        wide_input_size = 0
        deep_input_size = 0

        for feature in self.features.number_features:
            if feature.name in self.wide_features:
                wide_input_size += 1
            if feature.name in self.deep_features:
                deep_input_size += 1

        for feature in self.features.category_features:
            if feature.name in self.wide_features:
                wide_input_size += self.embedding_sizes[feature.name]
            if feature.name in self.deep_features:
                deep_input_size += self.embedding_sizes[feature.name]

        for feature in self.features.sequence_features:
            self._sequence_poolings[feature.name] = SumPooling(1)
            self.add_module(
                f"pooling:{feature.name}",
                self._sequence_poolings[feature.name])
            if feature.name in self.wide_features:
                wide_input_size += self.embedding_sizes[feature.name]
            if feature.name in self.deep_features:
                deep_input_size += self.embedding_sizes[feature.name]

        # plus cross embedding size
        wide_input_size += len(self.cross_features) * embedding_size

        final_layer_input_size = wide_input_size

        if deep_input_size:
            self.mlp = MLP(
                deep_input_size,
                hidden_layers,
                dropout=dropout, batchnorm=True, activation=activation)
            final_layer_input_size += hidden_layers[-1]

        output_size = self.num_classes

        if self.num_classes == 2 and self.final_activation == 'sigmoid':
            output_size -= 1

        self.final_layer = nn.Linear(final_layer_input_size, output_size)

        self.apply(init_weights)

    def forward(self, x):
        wide_inputs = list()
        deep_inputs = list()
        cross_inputs = list()

        for feature in self.features.number_features:
            if feature.name in self.wide_features:
                wide_inputs.append(x[feature.name].view(-1, 1))
            if feature.name in self.deep_features:
                deep_inputs.append(x[feature.name].view(-1, 1))

        for feature in self.features.category_features:
            if feature.name in self.wide_features:
                wide_inputs.append(
                    self.embeddings[feature.name](x[feature.name]))
            if feature.name in self.deep_features:
                deep_inputs.append(
                    self.embeddings[feature.name](x[feature.name]))

        for feature in self.features.sequence_features:
            if feature.name in self.wide_features:
                wide_inputs.append(
                    self._sequence_poolings[feature.name](
                        self.embeddings[feature.name](
                            x[feature.name])))
            if feature.name in self.deep_features:
                deep_inputs.append(
                    self._sequence_poolings[feature.name](
                        self.embeddings[feature.name](
                            x[feature.name])))

        # prepare cross features
        for x_f, y_f in self.cross_features:
            if x_f in self._sequence_poolings:
                x_emb = self._sequence_poolings[x_f](
                    self.embeddings[x_f](x[x_f]))
            else:
                x_emb = self.embeddings[x_f](x[x_f])

            if y_f in self._sequence_poolings:
                y_emb = self._sequence_poolings[y_f](
                    self.embeddings[y_f](x[y_f]))
            else:
                y_emb = self.embeddings[y_f](x[y_f])
            cross_inputs.append(x_emb * y_emb)

        final_layer_inputs = list()
        if wide_inputs:
            final_layer_inputs.append(torch.cat(wide_inputs, dim=1))

        if cross_inputs:
            final_layer_inputs.append(torch.cat(cross_inputs, dim=1))

        if deep_inputs:
            final_layer_inputs.append(self.mlp(torch.cat(deep_inputs, dim=1)))

        output = self.final_layer(torch.cat(final_layer_inputs, dim=1))

        if self.num_classes == 2 and self.final_activation == 'sigmoid':
            output = torch.sigmoid(output)
        elif self.num_classes > 1 and self.final_activation == 'softmax':
            output = torch.softmax(output)
        elif self.final_activation:
            raise NotImplementedError(
                f"pair (final_activation: {self.final_activation}, "
                f"num_classes: {self.num_classes}) is not implemented")

        return output
