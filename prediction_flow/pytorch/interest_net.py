"""Interest Net.
"""

from collections import OrderedDict

import torch
import torch.nn as nn

from .base import EmbeddingMixin
from .nn import MLP, MaxPooling
from .utils import init_weights


class AttentionGroup(object):
    """ This class is used to identify which features should be
    processed by attention. All candidate features and all behavior
    sequential features must be the same embedding size. All behavior
    sequential features must be the same maximum length.

    Parameters
    ----------
    name : str
        Unique group name.

    pairs : list of dict
        Example :
            [{'ad': 'item_id',
              'pos_hist': 'clicked_item_ids',
              'neg_hist': 'neg_item_ids'},
             {'ad': 'item_category',
              'pos_hist': 'clicked_item_categories',
              'neg_hist': 'neg_item_categories'}]

    hidden_layers : iterable
        Hidden layer sizes of attention.

    activation : str
        Activation function of attention.
        Example: prelu

    att_dropout : float
        Dropout rate of attention.

    gru_type : str
        Type of GRU. GRU, AIGRU, AGRU and AUGRU are supported.

    gru_dropout : float
        Dropout rate of GRU.
    """
    def __init__(self, name, pairs,
                 hidden_layers, activation='prelu', att_dropout=0.0,
                 gru_type='GRU', gru_dropout=0.0):
        self.name = name
        self.pairs = pairs
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.att_dropout = att_dropout
        self.gru_type = gru_type
        self.gru_dropout = gru_dropout

        self.related_feature_names = set()
        self.neg_feature_names = set()
        for pair in pairs:
            self.related_feature_names.add(pair['ad'])
            self.related_feature_names.add(pair['pos_hist'])
            if 'neg_hist' in pair:
                self.related_feature_names.add(pair['neg_hist'])
                self.neg_feature_names.add(pair['neg_hist'])

    def is_attention_feature(self, feature_name):
        if feature_name in self.related_feature_names:
            return True
        return False

    def is_neg_sampling_feature(self, feature_name):
        if feature_name in self.neg_feature_names:
            return True
        return False

    @property
    def pairs_count(self):
        return len(self.pairs)


class InterestNet(nn.Module, EmbeddingMixin):
    """Interest Network.

    Parameters
    ----------
    features : Features

    attention_groups : list of AttentionGroup

    num_classes : int
        Number of classes.

    embedding_size : int
        Size of embedding.

    hidden_layers : list
        Size of hidden layers.
        Example: [96, 32]

    dnn_activation : str
        Activation function of deep layers.
        Example: relu

    final_activation : str
        Activation function of output.

    dropout : float
        Dropout rate.
    """
    def _is_attention_feature(self, feature):
        for group in self.attention_groups:
            if group.is_attention_feature(feature.name):
                return True
        return False

    def _is_neg_sampling_feature(self, feature):
        for group in self.attention_groups:
            if group.is_neg_sampling_feature(feature.name):
                return True
        return False

    def create_attention_fn(self, attention_group):
        raise NotImplementedError(
            "Please implement the func to create attention")

    def __init__(self, features, attention_groups, num_classes, embedding_size,
                 hidden_layers, dnn_activation='prelu', final_activation=None,
                 dropout=0.0):
        super(InterestNet, self).__init__()
        self.features = features
        self.attention_groups = attention_groups
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.hidden_layers = hidden_layers
        self.dnn_activation = dnn_activation
        self.final_activation = final_activation
        self.dropout = dropout

        self.embeddings, self.embedding_sizes = self.build_embeddings(
            embedding_size)

        self._sequence_poolings = OrderedDict()
        self._attention_poolings = OrderedDict()

        total_embedding_sizes = 0
        for feature in self.features.category_features:
            total_embedding_sizes += (
                self.embedding_sizes[feature.name])

        for feature in self.features.sequence_features:
            if not self._is_neg_sampling_feature(feature):
                total_embedding_sizes += (
                    self.embedding_sizes[feature.name])
            if not self._is_attention_feature(feature):
                self._sequence_poolings[feature.name] = MaxPooling(1)
                self.add_module(
                    f"pooling:{feature.name}",
                    self._sequence_poolings[feature.name])

        # attention
        for attention_group in self.attention_groups:
            self._attention_poolings[attention_group.name] = (
                self.create_attention_fn(attention_group))
            self.add_module(
                f"attention_pooling:{attention_group.name}",
                self._attention_poolings[attention_group.name])

        total_input_size = (total_embedding_sizes +
                            len(self.features.number_features))
        self.mlp = MLP(
            total_input_size,
            hidden_layers,
            dropout=dropout, batchnorm=True, activation=dnn_activation)
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

        embeddings = OrderedDict()
        for feature in self.features.category_features:
            embeddings[feature.name] = self.embeddings[
                feature.name](x[feature.name])

        for feature in self.features.sequence_features:
            if not self._is_attention_feature(feature):
                embeddings[feature.name] = self._sequence_poolings[
                    feature.name](self.embeddings[
                        feature.name](x[feature.name]))

        for attention_group in self.attention_groups:
            query = torch.cat(
                [embeddings[pair['ad']]
                 for pair in attention_group.pairs],
                dim=-1)
            keys = torch.cat(
                [self.embeddings[pair['pos_hist']](
                    x[pair['pos_hist']]) for pair in attention_group.pairs],
                dim=-1)
            keys_length = torch.min(torch.cat(
                [x[f"__{pair['pos_hist']}_length"].view(-1, 1)
                 for pair in attention_group.pairs],
                dim=-1), dim=-1)[0]
            embeddings[attention_group.name] = self._attention_poolings[
                attention_group.name](query, keys, keys_length)

        emb_concat = torch.cat(number_inputs + [
            emb for emb in embeddings.values()], dim=-1)

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
