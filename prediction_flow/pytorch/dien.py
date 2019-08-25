"""
Deep Interest Evolution Network.
"""

from collections import OrderedDict

import torch

from .nn import Interest
from .interest_net import InterestNet


class DIEN(InterestNet):
    """Deep Interest Evolution Network.

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

    use_negsampling : bool
    """
    def __init__(self, *args, use_negsampling=False, **kwargs):
        self.use_negsampling = use_negsampling
        super(DIEN, self).__init__(*args, **kwargs)

    def create_attention_fn(self, attention_group):
        return Interest(
            attention_group.pairs_count * self.embedding_size,
            gru_type=attention_group.gru_type,
            gru_dropout=attention_group.gru_dropout,
            att_hidden_layers=attention_group.hidden_layers,
            att_dropout=attention_group.att_dropout,
            att_activation=attention_group.activation,
            use_negsampling=self.use_negsampling)

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

        auxiliary_losses = []
        for attention_group in self.attention_groups:
            query = torch.cat(
                [embeddings[pair['ad']]
                 for pair in attention_group.pairs],
                dim=-1)
            pos_hist = torch.cat(
                [self.embeddings[pair['pos_hist']](
                    x[pair['pos_hist']]) for pair in attention_group.pairs],
                dim=-1)
            keys_length = torch.min(torch.cat(
                [x[f"__{pair['pos_hist']}_length"].view(-1, 1)
                 for pair in attention_group.pairs],
                dim=-1), dim=-1)[0]
            neg_hist = None
            if self.use_negsampling:
                neg_hist = torch.cat(
                    [self.embeddings[pair['neg_hist']](
                        x[pair['neg_hist']])
                     for pair in attention_group.pairs],
                    dim=-1)
            embeddings[attention_group.name], tmp_loss = (
                self._attention_poolings[attention_group.name](
                    query, pos_hist, keys_length, neg_hist))
            if tmp_loss is not None:
                auxiliary_losses.append(tmp_loss)

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

        auxiliary_avg_loss = None
        if auxiliary_losses:
            auxiliary_avg_loss = auxiliary_losses[0]
            size = len(auxiliary_losses)
            for i in range(1, size):
                auxiliary_avg_loss += auxiliary_losses[i]
            auxiliary_avg_loss /= size

        return output, auxiliary_avg_loss
