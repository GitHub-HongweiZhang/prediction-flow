from prediction_flow.features import Number, Category, Sequence, Features
from prediction_flow.transformers.column import (
    StandardScaler, CategoryEncoder, SequenceEncoder)
from prediction_flow.pytorch.data import Dataset

import pandas as pd


__SAMPLE_DF = pd.DataFrame({
    'userId': [11, 11, 11, 11, 11],
    'userAge': [23, 21, 19, 17, 41],
    'movieId': [4226, 5971, 6291, 7153, 30707],
    'rating': [3.0, 2.0, 4.0, 4.6, 5.0],
    'timestamp': [1294796159, 1294796201, 1294796113, 1294796132, 1294796176],
    'title': ['Memento (2000)',
              'My Neighbor Totoro (Tonari no Totoro) (1988)',
              'Lilya 4-Ever (Lilja 4-ever) (2002)',
              'Lord of the Rings: The Return of the King, The (2003)',
              'Million Dollar Baby (2004)'],
    'genres': [
        'Mystery|Thriller',
        'Animation|Children|Drama|Fantasy',
        'Crime|Drama',
        'Action|Adventure|Drama|Fantasy',
        'Drama'],
    'topGenre': [
        'Mystery',
        'Animation',
        'Crime',
        'Action',
        'Drama'],
    'clickedMovieIds': [
        '5971|6291',
        '3242|42',
        '32|43542|3222|3',
        '',
        '34|23'],
    'clickedMovieTopGenres': [
        'Animation|Mystery',
        'Drama',
        'Drama',
        '',
        'Mystery|Crime'],
    'label': [1, 0, 0, 1, 0]})


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

    features.fit(__SAMPLE_DF)

    X_map = features.transform(__SAMPLE_DF)

    dataset = Dataset(features, X_map, __SAMPLE_DF.label.values)

    assert dataset[0]['userId'] == 1
    assert dataset[0]['movieId'] == 1
    assert dataset[0]['genres'].tolist() == [8, 9, 0, 0]
    assert dataset[0]['__genres_length'] == 2
    assert dataset[0]['label'] == 1
