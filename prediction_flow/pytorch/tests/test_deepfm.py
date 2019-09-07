from prediction_flow.features import Number, Category, Sequence, Features
from prediction_flow.transformers.column import (
    StandardScaler, CategoryEncoder, SequenceEncoder)
from prediction_flow.pytorch import DeepFM

from .utils import prepare_dataloader


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

    deep_fm = DeepFM(
        features, num_classes=2, embedding_size=4, hidden_layers=(8, 4),
        final_activation='sigmoid', dropout=0.3)

    deep_fm(next(iter(dataloader)))


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

    deep_fm = DeepFM(
        features, num_classes=2, embedding_size=4, hidden_layers=(8, 4),
        final_activation='sigmoid', dropout=0.3)

    deep_fm(next(iter(dataloader)))


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

    deep_fm = DeepFM(
        features, num_classes=2, embedding_size=4, hidden_layers=(8, 4),
        final_activation='sigmoid', dropout=0.3)

    deep_fm(next(iter(dataloader)))


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

    deep_fm = DeepFM(
        features, num_classes=2, embedding_size=4, hidden_layers=(8, 4),
        final_activation='sigmoid', dropout=0.3)

    deep_fm(next(iter(dataloader)))
