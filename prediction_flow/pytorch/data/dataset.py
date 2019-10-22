"""Dataset for torch.
"""

# Authors: Hongwei Zhang
# License: MIT


from collections import OrderedDict
from itertools import chain

import numpy as np
import torch.utils.data as data


class Dataset(data.Dataset):
    """Dataset for torch.

    Parameters
    ----------
    features : Features
        Fitted Features object.

    X_map : dict
        example:
            {'feature1': numpy.array([...]), 'feature2': numpy.array([...])}

    y : numpy.array
    """
    def __init__(self, features, X_map, y=None):
        self.features = features
        self.X_map = X_map
        self.y = y
        if y is not None:
            self.y = np.asarray(y, np.float32).reshape(-1, 1)

        self.__data_size = self.__get_data_size()

    def __get_data_size(self):
        key = next(iter(self.X_map))
        return self.X_map[key].shape[0]

    def __len__(self):
        return self.__data_size

    @staticmethod
    def __pad_sequence(sequence_feature, sequence):
        # zero is special index for padding
        padded_seq = np.zeros(sequence_feature.max_length(), np.int64)
        padded_seq[0: sequence.shape[0]] = sequence

        return padded_seq

    def __getitem__(self, idx):
        record = OrderedDict()

        for feat in chain(
                self.features.number_features,
                self.features.category_features):
            record[feat.name] = self.X_map[feat.name][idx]

        for feat in self.features.sequence_features:
            seq = self.X_map[feat.name][idx]
            record[feat.name] = Dataset.__pad_sequence(feat, seq)
            record[f"__{feat.name}_length"] = np.int64(seq.shape[0])

        if self.y is not None:
            record['label'] = self.y[idx]
        return record

    def get_num_batches(self, batch_size):
        return np.ceil(self.__data_size / batch_size)
