"""
DeepFM.
"""
from collections import OrderedDict

import torch
import torch.nn as nn

from .base import EmbeddingMixin
from .nn import FM, MLP, MaxPooling
from .utils import init_weights


class DeepFM(nn.Module, EmbeddingMixin):
    """DeepFM.

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

    use_linear : bool

    use_fm : bool

    use_deep : bool
    """
    def __init__(self, features, num_classes, embedding_size, hidden_layers,
                 activation='relu', final_activation=None, dropout=None,
                 use_linear=True, use_fm=True, use_deep=True):
        super(DeepFM, self).__init__()
        self.features = features
        self.num_classes = num_classes
        self.final_activation = final_activation
        self.use_linear = use_linear
        self.use_fm = use_fm
        self.use_deep = use_deep

        self.embeddings, self.embedding_sizes = self.build_embeddings(
            embedding_size, fixed_embedding_size=True)

        self._sequence_poolings = OrderedDict()

        total_embedding_sizes = 0
        for feature in self.features.number_features:
            self.embeddings[feature.name] = nn.Linear(
                in_features=1,
                out_features=embedding_size,
                bias=False)
            self.add_module(
                f"embedding:{feature.name}",
                self.embeddings[feature.name])
            total_embedding_sizes += embedding_size

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

        final_layer_input_size = 0
        # linear
        # This part is diff from deepFM paper,
        # sparse features are not included.
        self.linear = None
        if self.use_linear and self.features.number_features:
            self.linear = MLP(
                len(self.features.number_features),
                hidden_layers=[1], dropout=None, activation=None)
            final_layer_input_size += 1

        # fm
        self.fm = None
        if use_fm and total_embedding_sizes:
            self.fm = FM()
            final_layer_input_size += 1

        # deep
        self.mlp = None
        if use_deep and total_embedding_sizes:
            total_input_size = (total_embedding_sizes +
                                len(self.features.number_features))
            self.mlp = MLP(
                total_input_size, hidden_layers,
                dropout=dropout, activation=activation)
            final_layer_input_size += hidden_layers[-1]

        output_size = self.num_classes

        if self.num_classes == 2 and self.final_activation == 'sigmoid':
            output_size -= 1

        self.final_layer = nn.Linear(final_layer_input_size, output_size)

        self.apply(init_weights)

    def forward(self, x):
        final_layer_inputs = list()

        number_inputs = list()
        if self.linear:
            # linear
            for feature in self.features.number_features:
                number_inputs.append(x[feature.name].view(-1, 1))
            linear_concat = torch.cat(number_inputs, dim=1)
            final_layer_inputs.append(self.linear(linear_concat))

        embeddings = list()
        for feature in self.features.number_features:
            embeddings.append(
                self.embeddings[feature.name](
                    x[feature.name].view(-1, 1)).unsqueeze(1))
        for feature in self.features.category_features:
            embeddings.append(
                self.embeddings[feature.name](x[feature.name]).unsqueeze(1))
        for feature in self.features.sequence_features:
            embeddings.append(
                self._sequence_poolings[feature.name](
                    self.embeddings[feature.name](x[feature.name])).unsqueeze(1))

        emb_concat = None
        if embeddings:
            emb_concat = torch.cat(embeddings, dim=1)
            b, f, e = emb_concat.size()
            # fm
            if self.fm:
                final_layer_inputs.append(self.fm(emb_concat))
            emb_concat = emb_concat.view(b, f * e)

        # deep
        if self.mlp:
            deep_input = torch.cat(number_inputs + [emb_concat], dim=1)
            final_layer_inputs.append(self.mlp(deep_input))

        final_layer_inputs = torch.cat(final_layer_inputs, dim=1)

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
