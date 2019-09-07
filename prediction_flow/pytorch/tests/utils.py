from prediction_flow.pytorch.data import Dataset


import pandas as pd
import torch.utils.data as data


_SAMPLE_DF = pd.DataFrame({
    'userId': [11, 11, 11, 11, 11],
    'userAge': [23, 21, 19, 17, 41],
    'movieId': ['4226', '5971', '6291', '7153', '30707'],
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
        'Drama|Drama|Drama|Drama',
        '',
        'Mystery|Crime'],
    'noClickedMovieIds': [
        '233|291',
        '324|421',
        '325|3542|322|33',
        '',
        '45|48'],
    'noClickedMovieTopGenres': [
        'Drama|Crime',
        'Animation|Mystery',
        'Mystery|Animation|Crime|Drama',
        '',
        'Crime|Mystery'],
    'label': [1, 0, 0, 1, 0]})


def prepare_dataloader(features):
    features.fit(_SAMPLE_DF)

    X_map = features.transform(_SAMPLE_DF)

    dataset = Dataset(features, X_map, _SAMPLE_DF.label.values)

    dataloader = data.DataLoader(
        dataset, batch_size=_SAMPLE_DF.shape[0], shuffle=False)

    return dataloader, _SAMPLE_DF
