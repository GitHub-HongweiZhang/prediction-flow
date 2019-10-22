"""
SequenceEncoder to convert sequence terms to sequence number.
"""

# Authors: Hongwei Zhang
# License: MIT


from collections import Counter
import numpy as np

from .base import SequenceColumn


class SequenceEncoder(SequenceColumn):
    """Encoder for sequence type feature. Convert terms to numbers.
    First index is 1.

    Parameters
    ----------
    sep : str, default=' '
        Separator of input sequence.

    min_cnt : int, default=5
        Minimum count of term.

    max_len: int, default=None
        Maximum length of sequence. If none is given,
        the maximum length of training sequence will be used.

    word2idx : dict
        Mappings from term to index.

    idx2word : dict
        Mappings from index to term.

    Attributes
    ----------
    sep : str, default=' '
        Separator of input sequence.

    min_cnt : int, default=5
        Minimum count of term.

    max_len: int, default=None
        Maximum length of sequence. If none is given,
        the maximum length of training sequence will be used.

    word2idx : dict
        Mappings from term to index.

    idx2word : dict
        Mappings from index to term.
    """
    def __init__(self, sep=' ', min_cnt=5, max_len=None,
                 word2idx=None, idx2word=None):
        self.sep = sep
        self.min_cnt = min_cnt
        self.max_len = max_len

        self.word2idx = word2idx if word2idx else dict()
        self.idx2word = idx2word if idx2word else dict()

    def fit(self, x, y=None):
        """Fit this transformer.

        Parameters
        ----------
        x : array-like
            One column of training data.
        y : array-like, default=None
            Training targets.

        Returns
        -------
        self : SequenceEncoder
            This SequenceEncoder.
        """

        if not self.word2idx:
            counter = Counter()

            max_len = 0
            for sequence in np.array(x).ravel():
                words = sequence.split(self.sep)
                counter.update(words)
                max_len = max(max_len, len(words))

            if self.max_len is None:
                self.max_len = max_len

            # drop rare words
            words = sorted(
                list(filter(lambda x: counter[x] >= self.min_cnt, counter)))

            self.word2idx = dict(zip(words, range(1, len(words) + 1)))
            self.word2idx['__PAD__'] = 0
            if '__UNKNOWN__' not in self.word2idx:
                self.word2idx['__UNKNOWN__'] = len(self.word2idx)

        if not self.idx2word:
            self.idx2word = {
                index: word for word, index in self.word2idx.items()}

        if not self.max_len:
            max_len = 0
            for sequence in np.array(x).ravel():
                words = sequence.split(self.sep)
                max_len = max(max_len, len(words))
            self.max_len = max_len

        return self

    def transform(self, x):
        """Transform x by this fitted transformer.

        Parameters
        ----------
        x : array-like
            Column data to be transformed.

        Returns
        -------
        transformed_x : array-like
            Transformed data.
        """
        transformed_x = list()

        for sequence in np.asarray(x).ravel():
            words = list()
            for word in sequence.split(self.sep):
                try:
                    words.append(self.word2idx[word])
                except KeyError:
                    words.append(self.word2idx['__UNKNOWN__'])

            transformed_x.append(
                np.asarray(words[0:self.max_len], dtype=np.int64))

        return np.asarray(transformed_x, dtype=np.object)

    def dimension(self):
        """Number of unique terms.
        """
        return len(self.word2idx)

    def max_length(self):
        """Maximum length of one sequence.
        """
        return self.max_len
