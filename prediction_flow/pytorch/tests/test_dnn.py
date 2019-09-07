import numpy as np

from prediction_flow.features import Number, Category, Sequence, Features
from prediction_flow.transformers.column import (
    StandardScaler, CategoryEncoder, SequenceEncoder)
from prediction_flow.pytorch import DNN


from .utils import prepare_dataloader, _SAMPLE_DF


def test_normal():
    number_features = [
        Number('userAge', StandardScaler()),
        Number('rating', StandardScaler())]

    category_features = [
        Category('userId', CategoryEncoder(min_cnt=1)),
        Category('movieId', CategoryEncoder(min_cnt=1)),
        Category('topGenre', CategoryEncoder(min_cnt=1))]

    sequence_features = [
        Sequence('title', SequenceEncoder(sep='|', min_cnt=1)),
        Sequence('genres', SequenceEncoder(sep='|', min_cnt=1)),
        Sequence('clickedMovieIds',
                 SequenceEncoder(sep='|', min_cnt=1, max_len=5)),
        Sequence('clickedMovieTopGenres',
                 SequenceEncoder(sep='|', min_cnt=1, max_len=5))]

    features = Features(
        number_features=number_features,
        category_features=category_features,
        sequence_features=sequence_features)

    dataloader, _ = prepare_dataloader(features)

    model = DNN(
        features, num_classes=2, embedding_size=4, hidden_layers=(8, 4),
        final_activation='sigmoid', dropout=0.3)

    model(next(iter(dataloader)))


def test_without_number_feature():
    number_features = []

    category_features = [
        Category('userId', CategoryEncoder(min_cnt=1)),
        Category('movieId', CategoryEncoder(min_cnt=1)),
        Category('topGenre', CategoryEncoder(min_cnt=1))]

    sequence_features = [
        Sequence('title', SequenceEncoder(sep='|', min_cnt=1)),
        Sequence('genres', SequenceEncoder(sep='|', min_cnt=1)),
        Sequence('clickedMovieIds',
                 SequenceEncoder(sep='|', min_cnt=1, max_len=5)),
        Sequence('clickedMovieTopGenres',
                 SequenceEncoder(sep='|', min_cnt=1, max_len=5))]

    features = Features(
        number_features=number_features,
        category_features=category_features,
        sequence_features=sequence_features)

    dataloader, _ = prepare_dataloader(features)

    model = DNN(
        features, num_classes=2, embedding_size=4, hidden_layers=(8, 4),
        final_activation='sigmoid', dropout=0.3)

    model(next(iter(dataloader)))


def test_without_category_feature():
    number_features = []

    category_features = []

    sequence_features = [
        Sequence('title', SequenceEncoder(sep='|', min_cnt=1)),
        Sequence('genres', SequenceEncoder(sep='|', min_cnt=1)),
        Sequence('clickedMovieIds',
                 SequenceEncoder(sep='|', min_cnt=1, max_len=5)),
        Sequence('clickedMovieTopGenres',
                 SequenceEncoder(sep='|', min_cnt=1, max_len=5))]

    features = Features(
        number_features=number_features,
        category_features=category_features,
        sequence_features=sequence_features)

    dataloader, _ = prepare_dataloader(features)

    model = DNN(
        features, num_classes=2, embedding_size=4, hidden_layers=(8, 4),
        final_activation='sigmoid', dropout=0.3)

    model(next(iter(dataloader)))


def test_only_with_number_features():
    number_features = [
        Number('userAge', StandardScaler()),
        Number('rating', StandardScaler())]

    category_features = []

    sequence_features = []

    features = Features(
        number_features=number_features,
        category_features=category_features,
        sequence_features=sequence_features)

    dataloader, _ = prepare_dataloader(features)

    model = DNN(
        features, num_classes=2, embedding_size=4, hidden_layers=(8, 4),
        final_activation='sigmoid', dropout=0.3)

    model(next(iter(dataloader)))


def test_shared_embedding():
    number_features = []

    movie_enc = SequenceEncoder(sep='|', min_cnt=1, max_len=5)
    genre_enc = SequenceEncoder(sep='|', min_cnt=1, max_len=5)

    movie_enc.fit(
        np.concatenate((_SAMPLE_DF.clickedMovieIds.values,
                        _SAMPLE_DF.movieId.values), axis=None))

    genre_enc.fit(
        np.concatenate((_SAMPLE_DF.clickedMovieTopGenres.values,
                        _SAMPLE_DF.topGenre.values), axis=None))

    category_features = [
        Category('userId', CategoryEncoder(min_cnt=1)),
        Category('movieId',
                 CategoryEncoder(
                     min_cnt=1,
                     word2idx=movie_enc.word2idx,
                     idx2word=movie_enc.idx2word),
                 embedding_name='movieId'),
        Category('topGenre',
                 CategoryEncoder(
                     min_cnt=1,
                     word2idx=genre_enc.word2idx,
                     idx2word=genre_enc.idx2word),
                 embedding_name='topGenre', embedding_size=8)]

    sequence_features = [
        Sequence('title', SequenceEncoder(sep='|', min_cnt=1)),
        Sequence('genres', SequenceEncoder(sep='|', min_cnt=1)),
        Sequence('clickedMovieIds',
                 SequenceEncoder(
                     sep='|',
                     min_cnt=1,
                     max_len=5,
                     word2idx=movie_enc.word2idx,
                     idx2word=movie_enc.idx2word),
                 embedding_name='movieId'),
        Sequence('clickedMovieTopGenres',
                 SequenceEncoder(
                     sep='|',
                     min_cnt=1,
                     max_len=5,
                     word2idx=genre_enc.word2idx,
                     idx2word=genre_enc.idx2word),
                 embedding_name='topGenre', embedding_size=8)]

    features = Features(
        number_features=number_features,
        category_features=category_features,
        sequence_features=sequence_features)

    dataloader, _ = prepare_dataloader(features)

    model = DNN(
        features, num_classes=2, embedding_size=16, hidden_layers=(8, 4),
        final_activation='sigmoid', dropout=0.3)

    model(next(iter(dataloader)))
